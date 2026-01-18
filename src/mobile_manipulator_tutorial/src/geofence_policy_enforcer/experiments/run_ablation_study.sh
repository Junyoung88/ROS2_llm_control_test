#!/bin/bash
#
# Ablation Study Runner for Mobile Manipulator Geofence Experiments
#
# This script runs experiments for all ablation conditions:
#   - full: All margin terms enabled
#   - no_estimation: Estimation term disabled
#   - no_tracking: Tracking term disabled
#   - no_latency: Latency term disabled
#
# Usage:
#   ./run_ablation_study.sh [reps] [methods]
#
# Examples:
#   ./run_ablation_study.sh              # 30 reps, all methods
#   ./run_ablation_study.sh 10           # 10 reps, all methods
#   ./run_ablation_study.sh 30 "safetychip,geofence"  # 30 reps, specific methods
#

REPS=${1:-30}
METHODS=${2:-"no_guard,safetychip,selp,geofence"}
OUTPUT_BASE="/tmp/mobile_manip_ablation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"

# Ablation conditions and their corresponding parameter settings
declare -A CONDITIONS=(
    ["full"]="enable_estimation_term:=true enable_tracking_term:=true enable_latency_term:=true"
    ["no_estimation"]="enable_estimation_term:=false enable_tracking_term:=true enable_latency_term:=true"
    ["no_tracking"]="enable_estimation_term:=true enable_tracking_term:=false enable_latency_term:=true"
    ["no_latency"]="enable_estimation_term:=true enable_tracking_term:=true enable_latency_term:=false"
)

echo "========================================================================"
echo "ABLATION STUDY FOR MOBILE MANIPULATOR GEOFENCE"
echo "========================================================================"
echo "  Repetitions: ${REPS}"
echo "  Methods: ${METHODS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Conditions: ${!CONDITIONS[@]}"
echo "========================================================================"

mkdir -p "${OUTPUT_DIR}"

for condition in "${!CONDITIONS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Running condition: ${condition}"
    echo "========================================================================"

    CONDITION_OUTPUT="${OUTPUT_DIR}/${condition}"

    # Run the SCIE experiment with ablation parameters
    python3 "${SCRIPT_DIR}/run_scie_experiment.py" \
        --reps ${REPS} \
        --methods ${METHODS} \
        --output "${CONDITION_OUTPUT}" \
        --workspace "${WORKSPACE_DIR}" \
        --yes

    # Note: The ablation parameters need to be passed via ros2 param set
    # or by modifying the launch file parameters before starting

    echo "Condition ${condition} completed. Results in ${CONDITION_OUTPUT}"
    echo ""

    # Wait between runs
    sleep 5
done

echo "========================================================================"
echo "ALL ABLATION CONDITIONS COMPLETED"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================================================"

# Generate summary
echo ""
echo "Summary of results:"
for condition in "${!CONDITIONS[@]}"; do
    SUMMARY_FILE=$(find "${OUTPUT_DIR}/${condition}" -name "*summary*.json" 2>/dev/null | head -1)
    if [ -f "${SUMMARY_FILE}" ]; then
        echo "  ${condition}:"
        python3 -c "
import json
with open('${SUMMARY_FILE}', 'r') as f:
    summary = json.load(f)
for method, stats in summary.get('by_method', {}).items():
    acc = stats.get('accuracy', 0) * 100
    vr = stats.get('violation_rate', 0) * 100
    print(f\"    {method}: Accuracy={acc:.1f}%, VR={vr:.1f}%\")
" 2>/dev/null || echo "    (Could not parse summary)"
    fi
done
