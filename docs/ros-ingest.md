# Live ingest: ROS bus, robot swap

The mapping loop only sees `Frame` (`rgb`, `depth` meters, `K`, `T_wc`, `t`). ROS is how a camera/VSLAM process hands that over. `rclpy` runs **in the mapper process** (Humble Docker, Python 3.10 + torch). There is no unix-socket sidecar and no Jazzy spawn.

See [`../docker/README.md`](../docker/README.md), [`progress.md`](progress.md), [`online-graph-pddl-architecture.md`](online-graph-pddl-architecture.md).

## What to swap on a robot

Replace the **phone publisher**. Keep the mapper.

| Module | Phone | Robot |
| --- | --- | --- |
| `scripts/record3d_ros_publisher.py` | USB + ROS publish | **gone** |
| robot camera + VSLAM | — | **ROS publish** (this is the swap) |
| `slam/ros_frame_source.py` (`RosRgbDPoseSource`) | in-process subscribe → `Frame` | **same** |
| `slam/frame_ipc.py` (`LatestFrameSlot`) | drop-oldest slot | **same** |
| `slam/frame_source.py` + `r3d_stream_rerun_realtime_mapping.py` | `Frame` only | **same** |

YOLO, CLIP, and fuse never see ROS.

## Phone (Record3D USB)

One USB client. Do not run `ingest=usb` on the mapper at the same time as the publisher.

```text
record3d_ros_publisher.py        USB → ROS topics
        /record3d/color, depth, camera_info, pose
RosRgbDPoseSource                in-process rclpy
        LatestFrameSlot → Frame
r3d_stream_rerun_realtime_mapping.py   ingest=ros
```

USB-only (no ROS camera topics): omit `ingest=ros`, do not start the publisher. The mapper still serves `/conceptgraph/find` in-process.

```bash
# Docker — one container, two processes (default ingest=ros)
python3 conceptgraph/scripts/record3d_ros_publisher.py
# other shell in the same container:
cd /ws/conceptgraph
python3 slam/r3d_stream_rerun_realtime_mapping.py ingest=ros
```

USB-only (no ROS camera topics): `ingest=usb`, do not start the publisher. The mapper still serves `/conceptgraph/find` in-process. The container must see host `usbmuxd` via `/host-run` (see `docker-compose.yml`).

## Robot

```text
robot camera + VSLAM
        ROS topics                          remap hydra ros_*_topic onto the robot
RosRgbDPoseSource
        Frame
r3d_stream_rerun_realtime_mapping.py        ingest=ros
```

Hydra (`hydra_configs/r3d_stream_rerun_realtime_mapping.yaml`):

```yaml
ingest: ros
ros_rgb_topic: /record3d/color/image_raw
ros_depth_topic: /record3d/depth/image_raw
ros_camera_info_topic: /record3d/color/camera_info
ros_pose_topic: /record3d/pose
```

Point those four names at the robot. A topic ending in `/compressed` or `/compressedDepth` is `sensor_msgs/CompressedImage`; anything else is raw `Image`. HSR:

```yaml
ros_rgb_topic: /head_rgbd_sensor/rgb/image_rect_color/compressed
ros_depth_topic: /head_rgbd_sensor/depth_registered/image_rect_raw/compressedDepth
ros_camera_info_topic: /head_rgbd_sensor/rgb/camera_info
ros_pose_topic: /hsr/cam_pose
```

Do not use `/compressed` for depth (8-bit JPEG). Pose is `geometry_msgs/PoseStamped` (`T_wc` optical). On the HSR, `python3 /ws/scripts/report_camera_pose.py` publishes `/hsr/cam_pose`. RGB/depth/pose stamps will not match exactly; `ros_sync_slop` (default 0.1 s) uses ApproximateTime. Point hydra `ros_*_topic` at record3d or HSR as needed. The mapper command is unchanged: `ingest=ros`.

## Frame / ROS contract

Publishers (phone today, robot later) must provide:

| Field | Meaning |
| --- | --- |
| color | rectified BGR8 (or RGB8), or JPEG `CompressedImage` on `…/compressed` |
| depth | registered to color, meters (`32FC1`) or millimetres (`16UC1`), or PNG `compressedDepth` |
| `CameraInfo.k` | pinhole of that color image (distortion ignored — send undistorted/rectified images) |
| pose | `T_wc`: OpenCV optical axes (x right, y down, z forward), world ← camera. Not `camera_link` (x forward, z up). Phone node applies the ARKit Y/Z flip **before** publish. |
| QoS | BEST_EFFORT, keep-last 1 |

Phone stamps all four messages with the same `now()` so ExactTime would work. A robot will not: RGB/depth come from the camera clock, pose from TF. `ros_sync_slop > 0` (default 0.1) uses ApproximateTime. Set `ros_sync_slop: 0` for ExactTime.

## Drop-oldest

If the mapper is slower than the camera, drop stale frames — never block the producer.

- ROS: BEST_EFFORT keep-last 1.
- `LatestFrameSlot` overwrite; mapper `next()` is a pull (`get`), not a callback.
- USB (no ROS): Record3D SDK already overwrites its buffers.

## Realtime `find()` (service, not action)

CLIP lookup is a short request/response (encode text + cosine vs N objects). Use a **ROS service**, not an action.

`GroundingService.find()` lives in the mapper. It copies `clip_ft` + metadata under `map_lock`, then scores the copy so a merge cannot leak deleted ids. The mapping thread holds that lock only while it mutates the map. The service callback runs on the shared `rclpy` executor.

```text
ros2 service call /conceptgraph/find
        FindRosService                       in-process rclpy
GroundingService.find()                      mapping process
```

Interfaces are built into the Docker image (`/opt/conceptgraph_ws`). Rebuild on the bind mount if you change `.srv`:

```bash
cd /ws/ros
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select conceptgraph_interfaces
source /ws/ros/install/setup.bash
```

While the mapper is running (`find_enabled: true`):

```bash
python3 conceptgraph/scripts/find_query.py mug

ros2 service call /conceptgraph/find conceptgraph_interfaces/srv/Find \
  "{text: 'mug', k: 5, min_sim: 0.25, min_obs: 3}"
```

`k`, `min_sim`, or `min_obs` ≤ 0 means “use the mapper default”. Snapshot / PDDL is a later API.
