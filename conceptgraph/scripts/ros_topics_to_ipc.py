"""Jazzy-only: ROS RGB-D + pose → unix socket (pull).

Run by RosRgbDPoseSource via /usr/bin/python3. Conda must not import this file's rclpy path.
This sidecar stays on a robot; only the phone publisher goes away. See docs/ros-ingest.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Thread

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptgraph.slam.frame_ipc import (  # noqa: E402
    LatestFrameSlot,
    bind_unix_server,
    quaternion_to_rotation_matrix,
    read_msg,
    write_msg,
)


def sensor_qos_profile():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def image_msg_to_bgr(msg):
    import numpy as np

    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("bgr8", "8UC3"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
    if msg.encoding in ("rgb8",):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return rgb[:, :, ::-1].copy()
    raise ValueError(f"unsupported color encoding {msg.encoding}")


def image_msg_to_depth_m(msg):
    import numpy as np

    h, w = int(msg.height), int(msg.width)
    if msg.encoding in ("32FC1", "32FC"):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w).copy()
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def camera_info_to_K(msg):
    import numpy as np

    return np.asarray(msg.k, dtype=np.float32).reshape(3, 3)


def pose_msg_to_T(msg):
    import numpy as np

    p = msg.pose.position
    q = msg.pose.orientation
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_to_rotation_matrix([q.x, q.y, q.z, q.w]).astype(np.float32)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def header_stamp_sec(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--rgb-topic", default="/record3d/color/image_raw")
    parser.add_argument("--depth-topic", default="/record3d/depth/image_raw")
    parser.add_argument("--info-topic", default="/record3d/color/camera_info")
    parser.add_argument("--pose-topic", default="/record3d/pose")
    parser.add_argument("--node-name", default="conceptgraph_rgbd_sub")
    args = parser.parse_args()

    import rclpy
    from message_filters import Subscriber, TimeSynchronizer
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image

    slot = LatestFrameSlot()
    rclpy.init()
    node = rclpy.create_node(args.node_name)
    qos = sensor_qos_profile()
    rgb_sub = Subscriber(node, Image, args.rgb_topic, qos_profile=qos)
    depth_sub = Subscriber(node, Image, args.depth_topic, qos_profile=qos)
    info_sub = Subscriber(node, CameraInfo, args.info_topic, qos_profile=qos)
    pose_sub = Subscriber(node, PoseStamped, args.pose_topic, qos_profile=qos)
    sync = TimeSynchronizer([rgb_sub, depth_sub, info_sub, pose_sub], queue_size=2)

    def on_sync(rgb_msg, depth_msg, info_msg, pose_msg):
        try:
            rgb = image_msg_to_bgr(rgb_msg)
            depth = image_msg_to_depth_m(depth_msg)
            if depth.shape[:2] != rgb.shape[:2]:
                import cv2

                depth = cv2.resize(
                    depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            slot.put(
                {
                    "rgb": rgb,
                    "depth": depth,
                    "K": camera_info_to_K(info_msg),
                    "pose": pose_msg_to_T(pose_msg),
                    "t": header_stamp_sec(rgb_msg.header),
                }
            )
        except Exception as exc:
            print(f"ROS sidecar skip frame: {exc}")

    sync.registerCallback(on_sync)
    # Keep filters alive.
    _keep = (rgb_sub, depth_sub, info_sub, pose_sub, sync)

    server = bind_unix_server(args.socket)
    print(
        f"ROS sidecar node={args.node_name} socket={args.socket}\n"
        f"  rgb   {args.rgb_topic}\n"
        f"  depth {args.depth_topic}\n"
        f"  info  {args.info_topic}\n"
        f"  pose  {args.pose_topic}"
    )

    spin_thread = Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    conn, _ = server.accept()
    try:
        while rclpy.ok():
            cmd = read_msg(conn)
            if cmd != "get":
                write_msg(conn, None)
                continue
            payload = slot.get(timeout=1.0)
            write_msg(conn, payload)
    except (KeyboardInterrupt, ConnectionError, BrokenPipeError):
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
