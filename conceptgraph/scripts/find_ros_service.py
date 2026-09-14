"""Jazzy-only ROS2 service: /conceptgraph/find → mapper unix socket.

A service, not an action: CLIP lookup is a short request/response.
Conda never imports rclpy. Build conceptgraph_interfaces first:

    source /opt/ros/jazzy/setup.bash
    cd ros && colcon build --packages-select conceptgraph_interfaces

The mapper sources ros/install/setup.bash when it spawns this process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptgraph.slam.frame_ipc import connect_unix, read_msg, write_msg  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/conceptgraph_find.sock")
    parser.add_argument("--service", default="/conceptgraph/find")
    parser.add_argument("--node-name", default="conceptgraph_find")
    args = parser.parse_args()

    try:
        import rclpy
        from conceptgraph_interfaces.srv import Find
    except ImportError:
        print(
            "conceptgraph_interfaces not found. From the repo:\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            "  cd ros && colcon build --packages-select conceptgraph_interfaces\n"
            "Find still works without ROS: python conceptgraph/scripts/find_query.py mug",
            file=sys.stderr,
        )
        raise SystemExit(1)

    conn = connect_unix(args.socket, timeout=20.0)
    rclpy.init()
    node = rclpy.create_node(args.node_name)

    def on_find(request, response):
        write_msg(
            conn,
            {
                "cmd": "find",
                "text": request.text,
                "k": request.k,
                "min_sim": request.min_sim,
                "min_obs": request.min_obs,
            },
        )
        reply = read_msg(conn)
        response.ok = bool(reply.get("ok", False))
        response.message = str(reply.get("message", ""))
        response.generation = int(reply.get("generation", 0))
        response.json = json.dumps(reply.get("hits") or [])
        return response

    node.create_service(Find, args.service, on_find)
    print(f"Find service {args.service} → {args.socket}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
