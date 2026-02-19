#!/bin/bash
# cmd_vel guard detection latency benchmark runner
#
# Starts guard node standalone, runs latency measurement, kills guard.
# Usage: ./run_latency_benchmark.sh [--n 100]
#
# Output: experiment_results/gazebo_s1_s6/guard_latency.json

set -e

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
GEOFENCE_CONFIG="${WORKSPACE_DIR}/src/mobile_manipulator_tutorial/src/geofence_policy_enforcer/config/geofence.yaml"
GUARD_LOG="/tmp/guard_latency_bench.log"
N_SAMPLES=100
OUTPUT="${WORKSPACE_DIR}/experiment_results/gazebo_s1_s6/guard_latency.json"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n) N_SAMPLES="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " cmd_vel Guard Detection Latency Benchmark"
echo "=============================================="
echo "  workspace : ${WORKSPACE_DIR}"
echo "  geofence  : ${GEOFENCE_CONFIG}"
echo "  n_samples : ${N_SAMPLES}"
echo "  output    : ${OUTPUT}"
echo "=============================================="

# ── Source ROS2 + workspace ──
source /opt/ros/jazzy/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"

# ── Kill any existing guard ──
pkill -f 'cmd_vel_guard_node' 2>/dev/null || true
sleep 0.5

# ── Start guard node (standalone, no Gazebo needed) ──
echo ""
echo "[1/3] Starting cmd_vel guard node..."
ros2 run geofence_policy_enforcer cmd_vel_guard_node \
    --ros-args \
    -p safety_method:=geofence \
    -p input_topic:=/cmd_vel_nav \
    -p output_topic:=/cmd_vel \
    -p odom_topic:=/odom \
    -p geofence_config:="${GEOFENCE_CONFIG}" \
    -p simulation_horizon:=2.0 \
    -p simulation_steps:=20 \
    -p override_strategy:=stop \
    -p simulated_comm_latency_ms:=0.0 \
    > "${GUARD_LOG}" 2>&1 &
GUARD_PID=$!
echo "  guard PID: ${GUARD_PID}"

# ── Wait for guard to fully initialize ──
echo "[2/3] Waiting for guard to initialize (3s)..."
sleep 3

# Verify guard is alive
if ! kill -0 "${GUARD_PID}" 2>/dev/null; then
    echo "[ERROR] Guard node died! Check log: ${GUARD_LOG}"
    tail -20 "${GUARD_LOG}"
    exit 1
fi
echo "  guard is running"

# ── Run benchmark ──
echo "[3/3] Running latency benchmark (n=${N_SAMPLES})..."
echo ""
python3 "${WORKSPACE_DIR}/src/geofence_enforcer/experiments/measure_guard_latency.py" \
    --n "${N_SAMPLES}" \
    --output "${OUTPUT}"

BENCH_EXIT=$?

# ── Cleanup ──
echo ""
echo "Stopping guard node (PID ${GUARD_PID})..."
kill "${GUARD_PID}" 2>/dev/null || true
wait "${GUARD_PID}" 2>/dev/null || true
pkill -f 'cmd_vel_guard_node' 2>/dev/null || true

if [ "${BENCH_EXIT}" -eq 0 ]; then
    echo ""
    echo "Done. Results: ${OUTPUT}"
    echo ""
    # Pretty-print summary
    python3 -c "
import json, sys
with open('${OUTPUT}') as f:
    d = json.load(f)
print('=== Final Results ===')
for key, stat in d['results'].items():
    if stat:
        print(f'  {key}:')
        print(f'    mean={stat[\"mean_ms\"]:.2f}ms  std={stat[\"std_ms\"]:.2f}ms  '
              f'median={stat[\"median_ms\"]:.2f}ms  P99={stat[\"p99_ms\"]:.2f}ms  n={stat[\"n\"]}')
"
else
    echo "[ERROR] Benchmark failed (exit ${BENCH_EXIT})"
    echo "Guard log: ${GUARD_LOG}"
    exit 1
fi
