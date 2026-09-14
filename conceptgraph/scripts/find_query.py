"""Query the live mapper find() socket from the conda env.

    conda activate conceptgraph
    python conceptgraph/scripts/find_query.py mug
    python conceptgraph/scripts/find_query.py "red chair" --k 8 --min-sim 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptgraph.slam.frame_ipc import connect_unix, read_msg, write_msg


def main():
    parser = argparse.ArgumentParser(description="CLIP find() against the live mapper.")
    parser.add_argument("text", help="query phrase")
    parser.add_argument("--socket", default="/tmp/conceptgraph_find.sock")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-sim", type=float, default=0.25)
    parser.add_argument("--min-obs", type=int, default=3)
    args = parser.parse_args()

    conn = connect_unix(args.socket, timeout=2.0)
    write_msg(
        conn,
        {
            "cmd": "find",
            "text": args.text,
            "k": args.k,
            "min_sim": args.min_sim,
            "min_obs": args.min_obs,
        },
    )
    reply = read_msg(conn)
    conn.close()
    if not reply.get("ok", False):
        print(reply.get("message", "find failed"), file=sys.stderr)
        raise SystemExit(1)
    hits = reply.get("hits") or []
    print(f"generation={reply.get('generation')}  hits={len(hits)}  query={args.text!r}")
    if not hits:
        print("  (none)")
        return
    for i, hit in enumerate(hits, start=1):
        c = hit.get("center") or [float("nan")] * 3
        print(
            f"  {i}. sim={hit['sim']:.3f}  n={hit['n_obs']:3d}  "
            f"{hit['class_name']:20s}  "
            f"xyz=({c[0]:6.2f}, {c[1]:6.2f}, {c[2]:6.2f})  {hit['id'][:8]}"
        )


if __name__ == "__main__":
    main()
