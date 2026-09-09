"""Record3D USB → ROS2 topics (color, depth, camera_info, pose).

Jazzy rclpy is Python 3.12. The conceptgraph conda env is 3.10 and cannot
import it. This script therefore runs as two processes when launched from conda:

  conda 3.10  USB pump  --unix socket-->  /usr/bin/python3  ROS publisher

On a robot this file is not used; the robot publishes the same four topics.
See docs/ros-ingest.md.

From the conda env (no need to source ROS in this shell):

    conda activate conceptgraph
    python conceptgraph/scripts/record3d_ros_publisher.py
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptgraph.slam.frame_ipc import (
    bind_unix_server,
    connect_unix,
    read_msg,
    rotation_matrix_to_quaternion,
    spawn_jazzy_python,
    write_msg,
)


def sensor_qos_profile():
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def make_header(stamp, frame_id: str):
    from std_msgs.msg import Header

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def bgr_to_image_msg(rgb_bgr: np.ndarray, header):
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header = header
    msg.height, msg.width = int(rgb_bgr.shape[0]), int(rgb_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb_bgr, dtype=np.uint8).tobytes()
    return msg


def depth_to_image_msg(depth_m: np.ndarray, header):
    from sensor_msgs.msg import Image

    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    msg = Image()
    msg.header = header
    msg.height, msg.width = int(depth.shape[0]), int(depth.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = depth.tobytes()
    return msg


def k_to_camera_info(K: np.ndarray, width: int, height: int, header):
    from sensor_msgs.msg import CameraInfo

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    msg = CameraInfo()
    msg.header = header
    msg.width = int(width)
    msg.height = int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def T_to_pose_msg(T_wc: np.ndarray, header):
    from geometry_msgs.msg import PoseStamped

    q = rotation_matrix_to_quaternion(T_wc[:3, :3])
    msg = PoseStamped()
    msg.header = header
    msg.pose.position.x = float(T_wc[0, 3])
    msg.pose.position.y = float(T_wc[1, 3])
    msg.pose.position.z = float(T_wc[2, 3])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def publish_from_socket(args) -> None:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node("record3d_ros_publisher")
    qos = sensor_qos_profile()
    pub_rgb = node.create_publisher(Image, args.rgb_topic, qos)
    pub_depth = node.create_publisher(Image, args.depth_topic, qos)
    pub_info = node.create_publisher(CameraInfo, args.info_topic, qos)
    pub_pose = node.create_publisher(PoseStamped, args.pose_topic, qos)

    print(f"ROS publisher connecting to {args.from_socket}")
    conn = connect_unix(args.from_socket)
    print(
        f"Publishing  rgb={args.rgb_topic}  depth={args.depth_topic}  "
        f"info={args.info_topic}  pose={args.pose_topic}"
    )
    n = 0
    try:
        while rclpy.ok():
            payload = read_msg(conn)
            rgb = payload["rgb"]
            depth = payload["depth"]
            K = np.asarray(payload["K"])
            T_wc = np.asarray(payload["pose"])
            stamp = node.get_clock().now().to_msg()
            header = make_header(stamp, args.frame_id)
            h, w = rgb.shape[:2]
            pub_rgb.publish(bgr_to_image_msg(rgb, header))
            pub_depth.publish(depth_to_image_msg(depth, header))
            pub_info.publish(k_to_camera_info(K, w, h, header))
            pub_pose.publish(T_to_pose_msg(T_wc, header))
            n += 1
            if n == 1 or n % 60 == 0:
                node.get_logger().info(
                    f"published {n}  rgb={rgb.shape}  depth={depth.shape}  fx={K[0, 0]:.1f}"
                )
            rclpy.spin_once(node, timeout_sec=0.0)
    except (KeyboardInterrupt, ConnectionError):
        pass
    node.destroy_node()
    rclpy.shutdown()
    print(f"stopped after {n} frames")


def usb_pump(args, sock_path: str) -> None:
    from conceptgraph.utils.record3d_utils import DemoApp

    import torch

    server = bind_unix_server(sock_path)
    print(f"USB pump listening on {sock_path}, waiting for ROS publisher...")
    server.settimeout(20.0)
    try:
        conn, _ = server.accept()
    except socket.timeout as exc:
        raise SystemExit("ROS publisher did not connect. Is /opt/ros/jazzy installed?") from exc
    print("Connecting Record3D USB...")
    app = DemoApp()
    app.connect_to_device(dev_idx=args.dev_idx)
    n = 0
    try:
        while True:
            rgb, depth, K4, T_wc = app.get_frame_data()
            if rgb is None:
                continue
            K = K4.cpu().numpy()[:3, :3] if torch.is_tensor(K4) else np.asarray(K4)[:3, :3]
            write_msg(
                conn,
                {
                    "rgb": rgb,
                    "depth": depth,
                    "K": np.asarray(K, dtype=np.float32),
                    "pose": np.asarray(T_wc, dtype=np.float32),
                    "t": float(payload_time()),
                },
            )
            n += 1
            if n == 1 or n % 60 == 0:
                print(f"USB pump sent {n}  rgb={rgb.shape}  fx={float(K[0, 0]):.1f}")
    except (KeyboardInterrupt, ConnectionError, BrokenPipeError):
        pass
    print(f"USB pump stopped after {n} frames")
    try:
        os.unlink(sock_path)
    except OSError:
        pass


def payload_time() -> float:
    import time

    return time.time()


def parse_args():
    parser = argparse.ArgumentParser(description="Publish Record3D USB RGB-D + pose on ROS2.")
    parser.add_argument("--dev-idx", type=int, default=0)
    parser.add_argument("--frame-id", default="record3d_camera")
    parser.add_argument("--rgb-topic", default="/record3d/color/image_raw")
    parser.add_argument("--depth-topic", default="/record3d/depth/image_raw")
    parser.add_argument("--info-topic", default="/record3d/color/camera_info")
    parser.add_argument("--pose-topic", default="/record3d/pose")
    parser.add_argument(
        "--from-socket",
        default="",
        help="Jazzy child: read frames from this unix socket and publish ROS.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.from_socket:
        try:
            import rclpy  # noqa: F401
        except ImportError:
            print(
                "This --from-socket process needs system Python 3.12 + Jazzy rclpy.\n"
                "Do not run --from-socket inside conda.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        publish_from_socket(args)
        return

    try:
        import record3d  # noqa: F401
    except ImportError:
        print(
            "record3d not found. Activate conda env conceptgraph, then:\n"
            "  python conceptgraph/scripts/record3d_ros_publisher.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    sock_path = f"/tmp/conceptgraph_r3d_{os.getpid()}.sock"
    child_args = [
        "--from-socket",
        sock_path,
        "--frame-id",
        args.frame_id,
        "--rgb-topic",
        args.rgb_topic,
        "--depth-topic",
        args.depth_topic,
        "--info-topic",
        args.info_topic,
        "--pose-topic",
        args.pose_topic,
    ]
    print(
        "Starting Jazzy ROS publisher in /usr/bin/python3 "
        f"(conda is {sys.version_info[0]}.{sys.version_info[1]}, rclpy needs 3.12)."
    )
    child = spawn_jazzy_python(str(Path(__file__).resolve()), child_args)
    try:
        usb_pump(args, sock_path)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except Exception:
                child.kill()


if __name__ == "__main__":
    main()
