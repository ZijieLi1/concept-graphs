# Progress log

Architecture and phases live in [`online-graph-pddl-architecture.md`](online-graph-pddl-architecture.md). Live USB/ROS ingest (ROS bus, robot swap): [`ros-ingest.md`](ros-ingest.md). This file is what landed, in date order.

## 2026-09-21

Unix sockets and Jazzy sidecars removed. Humble Docker is the interpreter: `rclpy` is in-process with torch.

- **Publisher** is a single process (USB + Humble `rclpy`).
- **`RosRgbDPoseSource`** subscribes in the mapper; `LatestFrameSlot` still drop-oldest.
- **`FindRosService`** is `/conceptgraph/find` in the mapper. CLI: `python3 conceptgraph/scripts/find_query.py mug`.
- Deleted `ros_topics_to_ipc.py`, `find_ros_service.py`, `find_query_server.py`. `frame_ipc.py` is only the slot + quaternion helpers.

## 2026-09-14

Live CLIP `find()` while mapping. A **ROS service**, not an action — lookup is encode-text + cosine, not a long goal.

- `GroundingService` copies `clip_ft` + id/class/aabb under `map_lock`, then scores the copy.
- Mapper serves `/tmp/conceptgraph_find.sock`. CLI: `python conceptgraph/scripts/find_query.py mug`.
- Optional Jazzy sidecar `/conceptgraph/find` after `colcon build` of `ros/conceptgraph_interfaces`. See [`ros-ingest.md`](ros-ingest.md).

## 2026-09-09

ROS2 ingest for the live mapper. Record3D USB is either owned by this process (`ingest=usb`) or by a publisher node (`ingest=ros`). The mapper never blocks the camera: a depth-1 slot overwrites, the mapping thread is the only consumer.

Conda is Python 3.10; Jazzy `rclpy` is 3.12 — they do not share an interpreter. Unix sockets bridge that gap; ROS is the bus a robot replaces. Details: [`ros-ingest.md`](ros-ingest.md).

### Shipped

- **Publisher** `conceptgraph/scripts/record3d_ros_publisher.py` — conda USB pump → unix socket → Jazzy `/usr/bin/python3` publishes `/record3d/color/image_raw` (bgr8), `/record3d/depth/image_raw` (32FC1 meters, RGB-sized), `/record3d/color/camera_info`, `/record3d/pose` (`PoseStamped`, ARKit Y/Z flip already applied). Same stamp on all four. QoS BEST_EFFORT keep-last 1.
- **Sidecar** `scripts/ros_topics_to_ipc.py` — Jazzy ExactTime sync → `LatestFrameSlot` → unix socket. `RosRgbDPoseSource` only pulls; conda never imports `rclpy`.
- **Mapper** `r3d_stream_rerun_realtime_mapping.py` hydra `ingest=usb|ros`. USB connect still happens after CLIP load. One USB client: when `ingest=ros`, do not also run the mapper on USB.

```bash
conda activate conceptgraph
python conceptgraph/scripts/record3d_ros_publisher.py
# mapper: cd conceptgraph && python slam/r3d_stream_rerun_realtime_mapping.py ingest=ros
```

## 2026-09-03

Live USB mapping is usable in a house at **15+ fps** with YOLO26-seg. The old ~2 s/frame was not 3D fuse: open-vocab YOLOE at `detection_conf: 0.1` dumped tens of boxes, and CLIP ViT-H-14 ran on every crop *before* `filter_gobs`.

### Shipped

- **Live mapper = offline mapper.** `r3d_stream_rerun_realtime_mapping.py` uses `init_detector` / `run_detector`, skips CLIP on empty dets, no OpenAI edges (`make_edges: false`), merge unpacks `do_edges` correctly, `process_edges(..., frame_idx)`. USB still supplies per-frame RGB-D, K, pose.
- **Four detectors.** `yolo26_seg` (COCO native seg, no SAM) · `yoloe` (ScanNet200 prompts) · `yoloe_pf` (LVIS/Objects365) · `yolo11_sam`. Hydra: `detector=` / `detector.yaml`. Live default is `yolo26_seg` with high `detection_conf` (~0.78) so CLIP stays cheap.
- **Ctrl+C save.** Frame cap then auto-save; Ctrl+C finishes the current frame and prompts whether to write the map. USB connect happens *after* CLIP load so the stream is not starved during init.
- **Live K is per-frame (AF).** Record3D/ARKit updates `fx/fy` as focus changes; that is correct for unprojection. A recorded `.r3d` stores the same in `perFrameIntrinsicCoeffs`, but preprocess only writes a single `metadata["K"]` into `dataconfig.yaml`, and it still resizes RGB to 1440×1920 while leaving K at 720×960. Offline replay is the weaker path until that is fixed.
- **Debug tools.** `test_sctipts/record_3d_detect.py` (USB → boxes + conf histogram), `record_3d_stream.py`, `yolo_performance.py`. Open3D forced onto X11 so the viewer does not segfault on Wayland.

### Still open (see architecture phases)

Stream still writes an RGB JPEG every frame (`get_stream_data_out_path`). `find()` / `snapshot()` / PDDL are not started. Open-vocab is fine for labeling if we **cap boxes before CLIP** (conf / top-k), not after. Phase 0 is still RAM/I/O; the live *rate* budget is already met.
