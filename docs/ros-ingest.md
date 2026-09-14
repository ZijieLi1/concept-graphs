# Live ingest: ROS bus, unix sockets, robot swap

The mapping loop only sees `Frame` (`rgb`, `depth` meters, `K`, `T_wc`, `t`). ROS is how a camera/VSLAM process hands that over. Unix sockets are **not** the robot interface — they exist because conda Python 3.10 cannot import Jazzy’s `rclpy` (C extension built for system Python 3.12).

Do not `source /opt/ros/jazzy/setup.bash` inside the `conceptgraph` conda env. Conda never imports `rclpy`. Jazzy always runs as `/usr/bin/python3`.

Related: [`progress.md`](progress.md), [`online-graph-pddl-architecture.md`](online-graph-pddl-architecture.md).

## What to swap on a robot

Replace the **phone publisher**. Keep the mapper and the Jazzy sidecar.

| Module | Phone | Robot |
| --- | --- | --- |
| `scripts/record3d_ros_publisher.py` | USB + socket + ROS publish | **gone** |
| robot camera + VSLAM | — | **ROS publish** (this is the swap) |
| `scripts/ros_topics_to_ipc.py` | ROS subscribe → socket | **same** |
| `slam/ros_frame_source.py` (`RosRgbDPoseSource`) | socket pull → `Frame` | **same** |
| `slam/frame_ipc.py` | both sockets | **mapper sidecar only** |
| `slam/frame_source.py` + `r3d_stream_rerun_realtime_mapping.py` | `Frame` only | **same** |

YOLO, CLIP, and fuse never see ROS or sockets.

## Phone (Record3D USB)

Two sockets: Record3D lives in conda, ROS lives in system Python.

```text
record3d_ros_publisher.py   (conda 3.10: USB pump)
        unix socket                         frame_ipc.py
record3d_ros_publisher.py   (Jazzy 3.12: --from-socket)
        ROS topics                          /record3d/color, depth, camera_info, pose
ros_topics_to_ipc.py        (Jazzy sidecar, spawned by the mapper)
        unix socket                         frame_ipc.py
RosRgbDPoseSource           (conda mapper)
        Frame
r3d_stream_rerun_realtime_mapping.py
```

One USB client. Do not run `ingest=usb` on the mapper at the same time as the publisher.

```bash
# terminal 1 — from repo root, conda env, do not source Jazzy
conda activate conceptgraph
python conceptgraph/scripts/record3d_ros_publisher.py

# terminal 2
conda activate conceptgraph
cd conceptgraph
python slam/r3d_stream_rerun_realtime_mapping.py ingest=ros
```

USB-only (no ROS): omit `ingest=ros`, do not start the publisher.

## Robot

The left half is gone. The robot already publishes ROS from system Python.

```text
robot camera + VSLAM
        ROS topics                          remap hydra ros_*_topic onto the robot
ros_topics_to_ipc.py        (Jazzy sidecar)
        unix socket                         frame_ipc.py  (interpreter bridge only)
RosRgbDPoseSource
        Frame
r3d_stream_rerun_realtime_mapping.py
```

Hydra (`hydra_configs/r3d_stream_rerun_realtime_mapping.yaml`):

```yaml
ingest: ros
ros_rgb_topic: /record3d/color/image_raw
ros_depth_topic: /record3d/depth/image_raw
ros_camera_info_topic: /record3d/color/camera_info
ros_pose_topic: /record3d/pose
```

Point those four names at the robot. The mapper command is unchanged: `ingest=ros`.

The remaining socket is an interpreter bridge. If the mapper later runs on system Python 3.12, fold `ros_topics_to_ipc.py` into `RosRgbDPoseSource` and drop sockets.

## Frame / ROS contract

Publishers (phone today, robot later) must provide:

| Field | Meaning |
| --- | --- |
| color | rectified BGR8 (or RGB8) |
| depth | registered to color, meters (`32FC1`) or millimetres (`16UC1`) |
| `CameraInfo.k` | pinhole of that color image (distortion ignored — send undistorted/rectified images) |
| pose | `T_wc`: OpenCV optical axes (x right, y down, z forward), world ← camera. Not `camera_link` (x forward, z up). Phone node applies the ARKit Y/Z flip **before** publish. |
| QoS | mapper/sidecar: BEST_EFFORT, keep-last 1 |

Phone stamps all four messages with the same `now()` so ExactTime sync works. A robot will not: RGB/depth come from the camera clock, pose from VSLAM/TF. ExactTime will drop frames. Before a robot, switch the sidecar to ApproximateTime (or RGB-D sync + TF lookup at the RGB stamp) and drop if pose is too old.

## Drop-oldest

If the mapper is slower than the camera, drop stale frames — never block the producer.

- ROS: BEST_EFFORT keep-last 1.
- Sidecar: `LatestFrameSlot` overwrite; mapper `next()` is a pull (`get`), not a callback.
- USB (no ROS): Record3D SDK already overwrites its buffers.

## Realtime `find()` (service, not action)

CLIP lookup is a short request/response (encode text + cosine vs N objects). Use a **ROS service**, not an action.

`GroundingService.find()` lives in the conda mapper. It copies `clip_ft` + metadata under `map_lock`, then scores the copy so a merge cannot leak deleted ids. The mapping thread holds that lock only while it mutates the map.

```text
ros2 service call /conceptgraph/find     Jazzy find_ros_service.py
        unix socket                      /tmp/conceptgraph_find.sock
GroundingService.find()                  conda mapper
```

One-time interface build (system Python / Jazzy, not conda):

```bash
source /opt/ros/jazzy/setup.bash
cd /home/blinky/ZJ-WS/concept-graphs/ros
colcon build --packages-select conceptgraph_interfaces
```

While the mapper is running (`find_enabled: true`):

```bash
# conda — no Jazzy source
python conceptgraph/scripts/find_query.py mug

# Jazzy shell (source jazzy + ros/install/setup.bash)
ros2 service call /conceptgraph/find conceptgraph_interfaces/srv/Find \
  "{text: 'mug', k: 5, min_sim: 0.25, min_obs: 3}"
```

`k`, `min_sim`, or `min_obs` ≤ 0 means “use the mapper default”. Snapshot / PDDL is a later API.