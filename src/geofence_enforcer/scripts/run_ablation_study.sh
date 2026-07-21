#!/bin/bash
#
# Ablation Study Runner for SCIE-level Dynamic Geofence Experiments
#
# This script runs experiments for all ablation conditions:
#   - full: All margin terms enabled
#   - no_estimation: Estimation term disabled
#   - no_tracking: Tracking term disabled
#   - no_latency: Latency term disabled
#
# Usage:
#   ./run_ablation_study.sh [seeds] [headless]
#
# Examples:
#   ./run_ablation_study.sh              # 30 seeds, GUI
#   ./run_ablation_study.sh 50           # 50 seeds, GUI
#   ./run_ablation_study.sh 30 True      # 30 seeds, headless
#

SEEDS=${1:-30}
HEADLESS=${2:-False}
OUTPUT_BASE="/tmp/ablation_study_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"

# Ablation conditions and their corresponding parameter settings
declare -A CONDITIONS=(
    ["full"]="enable_estimation_term:=True enable_tracking_term:=True enable_latency_term:=True"
    ["no_estimation"]="enable_estimation_term:=False enable_tracking_term:=True enable_latency_term:=True"
    ["no_tracking"]="enable_estimation_term:=True enable_tracking_term:=False enable_latency_term:=True"
    ["no_latency"]="enable_estimation_term:=True enable_tracking_term:=True enable_latency_term:=False"
)

echo "========================================================================"
echo "ABLATION STUDY FOR DYNAMIC GEOFENCE EXPERIMENTS"
echo "========================================================================"
echo "  Seeds per condition: ${SEEDS}"
echo "  Headless: ${HEADLESS}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Conditions: ${!CONDITIONS[@]}"
echo "========================================================================"

mkdir -p "${OUTPUT_DIR}"

for condition in "${!CONDITIONS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Running condition: ${condition}"
    echo "========================================================================"

    CONDITION_OUTPUT="${OUTPUT_DIR}/${condition}"
    mkdir -p "${CONDITION_OUTPUT}"

    # Parse the parameter string
    PARAMS="${CONDITIONS[$condition]}"

    ros2 launch geofence_enforcer unified_gazebo_experiment.launch.py \
        seeds:=${SEEDS} \
        ablation_condition:=${condition} \
        use_dynamic_margin:=True \
        headless:=${HEADLESS} \
        output_dir:=${CONDITION_OUTPUT} \
        ${PARAMS}

    echo "Condition ${condition} completed. Results in ${CONDITION_OUTPUT}"
    echo ""

    # Wait a bit between runs to let Gazebo fully shutdown
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
    SUMMARY_FILE=$(find "${OUTPUT_DIR}/${condition}" -name "*summary*.json" | head -1)
    if [ -f "${SUMMARY_FILE}" ]; then
        echo "  ${condition}:"
        python3 -c "
import json
with open('${SUMMARY_FILE}', 'r') as f:
    summary = json.load(f)
for method, stats in summary.get('by_method', {}).items():
    print(f\"    {method}: SR={stats.get('SR', 0):.1f}%, VR={stats.get('VR', 0):.1f}%\")
"
    fi
done
