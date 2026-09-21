#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
if [[ -f /opt/conceptgraph_ws/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/conceptgraph_ws/install/setup.bash
fi
if [[ -f /ws/ros/install/setup.bash ]]; then
  # Bind-mount overlay wins when you rebuild Find.srv in /ws/ros.
  # shellcheck disable=SC1091
  source /ws/ros/install/setup.bash
fi
# Bind-mounting the usbmuxd socket *file* goes stale when the daemon
# exits (phone unplug) and recreates it. /host-run is the live host /run.
if [[ -d /host-run ]]; then
  mkdir -p /var/run
  ln -sfn /host-run/usbmuxd /var/run/usbmuxd
fi
export PYTHONPATH="/ws:${PYTHONPATH:-}"
cd /ws
exec "$@"
