#!/bin/bash
#
# S1-S7 Sequential Experiment Runner (Shell Wrapper)
# ==================================================
#
# 사용법:
#   ./run_experiments.sh                    # 기본 실험 실행
#   ./run_experiments.sh --quick            # 빠른 테스트 (S1만, 2개 method)
#   ./run_experiments.sh --all              # 전체 실험
#   ./run_experiments.sh --method geofence  # 특정 method만
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="/home/jim/ros2_motion_planning_tutorials/src/mobile_manipulator_tutorial"
OUTPUT_DIR="/tmp/s1_s7_results_$(date +%Y%m%d_%H%M%S)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         S1-S7 Sequential Experiment Runner                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick           Quick test (S1 only, geofence + safetychip)"
    echo "  --all             Run all scenarios with all methods"
    echo "  --method METHOD   Run specific method (no_guard, safetychip, selp, geofence)"
    echo "  --scenario S1     Run specific scenario"
    echo "  --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --quick                    # Fast test"
    echo "  $0 --all                      # Full experiment"
    echo "  $0 --method geofence          # Test geofence only"
    echo "  $0 --scenario S1 --method safetychip  # S1 with safetychip"
}

cleanup() {
    echo -e "\n${YELLOW}[INFO] Cleaning up...${NC}"
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ros2 launch" 2>/dev/null || true
    pkill -9 -f "goal_gate_node" 2>/dev/null || true
    pkill -9 -f "nav2" 2>/dev/null || true
    sleep 3
    echo -e "${GREEN}[INFO] Cleanup complete${NC}"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Parse arguments
SCENARIOS="S1 S2 S3 S4 S5 S6 S7"
METHODS="no_guard safetychip selp geofence"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            SCENARIOS="S1"
            METHODS="geofence safetychip"
            shift
            ;;
        --all)
            SCENARIOS="S1 S2 S3 S4 S5 S6 S7"
            METHODS="no_guard safetychip selp geofence"
            shift
            ;;
        --method)
            METHODS="$2"
            shift 2
            ;;
        --scenario)
            SCENARIOS="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

print_header

echo -e "${YELLOW}Configuration:${NC}"
echo "  Workspace: $WORKSPACE"
echo "  Output:    $OUTPUT_DIR"
echo "  Scenarios: $SCENARIOS"
echo "  Methods:   $METHODS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Source ROS2 workspace
echo -e "${BLUE}[INFO] Sourcing ROS2 workspace...${NC}"
source "$WORKSPACE/install/setup.bash"

# Initial cleanup
echo -e "${BLUE}[INFO] Initial cleanup...${NC}"
cleanup

# Run Python experiment script
echo -e "\n${GREEN}[INFO] Starting experiments...${NC}\n"

python3 "$SCRIPT_DIR/run_s1_s7_experiments.py" \
    --workspace "$WORKSPACE" \
    --output "$OUTPUT_DIR" \
    --scenarios $SCENARIOS \
    --methods $METHODS

echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Experiments Complete!                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
