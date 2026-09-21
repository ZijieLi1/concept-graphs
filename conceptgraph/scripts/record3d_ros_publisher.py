"""Record3D USB → ROS2 topics (color, depth, camera_info, pose).

One process: USB + rclpy. For Docker Humble / any env where record3d and
rclpy share an interpreter. On a robot this file is unused.

    python3 conceptgraph/scripts/record3d_ros_publisher.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptgraph.slam.frame_ipc import rotation_matrix_to_quaternion
from conceptgraph.utils.record3d_utils import DemoApp


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


def main():
    parser = argparse.ArgumentParser(description="Publish Record3D USB RGB-D + pose on ROS2.")
    parser.add_argument("--dev-idx", type=int, default=0)
    parser.add_argument("--frame-id", default="record3d_camera")
    parser.add_argument("--rgb-topic", default="/record3d/color/image_raw")
    parser.add_argument("--depth-topic", default="/record3d/depth/image_raw")
    parser.add_argument("--info-topic", default="/record3d/color/camera_info")
    parser.add_argument("--pose-topic", default="/record3d/pose")
    args = parser.parse_args()

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image
        import torch
    except ImportError as exc:
        print(f"missing import: {exc}", file=sys.stderr)
        raise SystemExit(1)

    rclpy.init()
    node = rclpy.create_node("record3d_ros_publisher")
    qos = sensor_qos_profile()
    pub_rgb = node.create_publisher(Image, args.rgb_topic, qos)
    pub_depth = node.create_publisher(Image, args.depth_topic, qos)
    pub_info = node.create_publisher(CameraInfo, args.info_topic, qos)
    pub_pose = node.create_publisher(PoseStamped, args.pose_topic, qos)

    app = DemoApp()
    print("Connecting Record3D USB...")
    app.connect_to_device(dev_idx=args.dev_idx)
    print(
        f"Publishing  rgb={args.rgb_topic}  depth={args.depth_topic}  "
        f"info={args.info_topic}  pose={args.pose_topic}"
    )
    n = 0
    try:
        while rclpy.ok():
            rgb, depth, K4, T_wc = app.get_frame_data()
            if rgb is None:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            K = K4.cpu().numpy()[:3, :3] if torch.is_tensor(K4) else np.asarray(K4)[:3, :3]
            T_wc = np.asarray(T_wc)
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
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    print(f"stopped after {n} frames")


if __name__ == "__main__":
    main()
