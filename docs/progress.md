# Progress log

Architecture and phases live in [`online-graph-pddl-architecture.md`](online-graph-pddl-architecture.md). This file is what landed, in date order.

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
