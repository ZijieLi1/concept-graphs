# Docker: ROS 2 Humble + mapping stack

Humble is Ubuntu 22.04 / **Python 3.10**, the same ABI as `record3d`. `rclpy` and PyTorch share one interpreter; unix-socket sidecars are gone.

Base: `osrf/ros:humble-desktop-full`. Torch 2.0.1 comes from the CUDA 11.8 wheel index; the host only needs a current NVIDIA driver and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Build and enter

```bash
# from repo root
xhost +local:docker   # Rerun / Open3D
docker compose build
docker compose run --rm conceptgraph
```

Inside the container ROS Humble and `conceptgraph_interfaces` are already sourced.

Default live ingest is ROS: two processes in **this** container (phone publisher + mapper). One USB client — the publisher.

```bash
# process 1 — USB + /record3d/* topics (Record3D USB streaming on the phone)
python3 conceptgraph/scripts/record3d_ros_publisher.py

# process 2 — same container, other shell (or background the publisher with &)
cd conceptgraph
python3 slam/r3d_stream_rerun_realtime_mapping.py ingest=ros
```

Another shell in the same container, from the host:

```bash
docker exec -it $(docker ps --filter ancestor=conceptgraph:humble -q | head -1) bash
```

USB mapping (phone in the mapper process): `ingest=usb`. Do not also run the publisher.

The iPhone is found via host `usbmuxd`. Compose mounts host `/run` at `/host-run` (a file bind-mount of the socket goes stale when usbmuxd restarts on unplug). Recreate the container after compose changes.

CLIP weights (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, ~4 GB) live in the Hugging Face hub cache. Compose bind-mounts `${HOME}/.cache/huggingface` so `docker compose run --rm` does not re-download.

Query the live map:

```bash
python3 conceptgraph/scripts/find_query.py mug
```

`nvidia-smi` should list the GPU. `python3 -c "import torch; print(torch.cuda.is_available())"` should be true when the toolkit is installed.

## Notes

- The repo is bind-mounted at `/ws`. Python edits on the host are live.
- `privileged` + `/dev/bus/usb` is for the iPhone. A robot that only publishes ROS topics does not need that.
- `network_mode: host` is for DDS on the same machine as other ROS nodes. Humble ↔ host Jazzy is not a supported mix; talk to other Humble nodes, or stay inside this container.
- `Find.srv` is built at image time into `/opt/conceptgraph_ws`. If you change the interface, rebuild under `/ws/ros` (entrypoint sources that overlay when present).
