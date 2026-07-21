#!/usr/bin/env bash
# Isolate the dense-reconstruction pipeline on its own ROS master.
#
# This machine runs other ROS nodes on the default master (:11311). Everything
# in the dense pipeline is developed and tested against :11390 instead, so a
# stray `rosnode kill -a`, a duplicate node name, or a topic name clash can
# never touch whatever else is live.
#
# Usage:
#   source scripts/dense_env.sh          # point this shell at the isolated master
#   source scripts/dense_env.sh --core   # ...and start a roscore there if none is up
#
# Check with: echo $ROS_MASTER_URI   /   rosnode list

export DENSE_ROS_PORT="${DENSE_ROS_PORT:-11390}"
export ROS_MASTER_URI="http://localhost:${DENSE_ROS_PORT}"
export ROS_HOSTNAME=localhost
# keep logs off the default master's pile too
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/dense_ros_log_${DENSE_ROS_PORT}}"
mkdir -p "$ROS_LOG_DIR"

echo "ROS_MASTER_URI=$ROS_MASTER_URI  (default :11311 untouched)"

if [ "$1" = "--core" ]; then
    if rostopic list >/dev/null 2>&1; then
        echo "roscore already up on :${DENSE_ROS_PORT}"
    else
        echo "starting roscore on :${DENSE_ROS_PORT}"
        roscore -p "${DENSE_ROS_PORT}" >"$ROS_LOG_DIR/roscore.log" 2>&1 &
        for _ in $(seq 1 30); do
            rostopic list >/dev/null 2>&1 && break
            sleep 0.5
        done
        rostopic list >/dev/null 2>&1 \
            && echo "roscore ready" \
            || echo "roscore FAILED to start; see $ROS_LOG_DIR/roscore.log"
    fi
fi
