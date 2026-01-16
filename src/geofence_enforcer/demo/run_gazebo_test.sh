#!/bin/bash
# =============================================================================
# Gazebo Integration Test Runner
# =============================================================================
#
# 이 스크립트는 Gazebo + Nav2 환경에서 Geofence Ablation Study를 실행합니다.
#
# 사전 요구사항:
#   1. ROS2 Humble 설치
#   2. TurtleBot3 패키지 설치:
#      sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-navigation2
#   3. Nav2 설치:
#      sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
#   4. geofence_enforcer 패키지 빌드:
#      cd ~/ros2_motion_planning_tutorials && colcon build --packages-select geofence_enforcer
#
# 사용법:
#   ./run_gazebo_test.sh [trials] [output_dir]
#
# 예시:
#   ./run_gazebo_test.sh          # 기본값: 10 trials
#   ./run_gazebo_test.sh 20       # 20 trials
#   ./run_gazebo_test.sh 20 /tmp/my_results
#
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 파라미터
TRIALS=${1:-10}
OUTPUT_DIR=${2:-/tmp/geofence_gazebo_test}

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       Geofence Gazebo Integration Test                     ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}Parameters:${NC}"
echo "  Trials per config: $TRIALS"
echo "  Output directory:  $OUTPUT_DIR"
echo ""

# ROS2 환경 확인
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}Error: ROS2 environment not sourced!${NC}"
    echo "Run: source /opt/ros/humble/setup.bash"
    exit 1
fi

echo -e "${GREEN}ROS2 $ROS_DISTRO detected${NC}"

# 워크스페이스 환경 설정
WORKSPACE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd )"
if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
    echo -e "${GREEN}Workspace sourced: $WORKSPACE_DIR${NC}"
else
    echo -e "${YELLOW}Warning: Workspace not built. Building now...${NC}"
    cd $WORKSPACE_DIR
    colcon build --packages-select geofence_enforcer
    source "$WORKSPACE_DIR/install/setup.bash"
fi

# TurtleBot3 모델 설정
export TURTLEBOT3_MODEL=burger
echo -e "${GREEN}TurtleBot3 model: $TURTLEBOT3_MODEL${NC}"

# 출력 디렉토리 생성
mkdir -p $OUTPUT_DIR

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       Starting Test                                        ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}This will:${NC}"
echo "  1. Launch Gazebo with TurtleBot3"
echo "  2. Launch Nav2 Navigation Stack"
echo "  3. Run Ablation Study (5 configs × $TRIALS trials = $((5 * TRIALS)) total)"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Launch 실행
ros2 launch geofence_enforcer gazebo_integration_test.launch.py \
    trials:=$TRIALS \
    output_dir:=$OUTPUT_DIR

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}       Test Completed!                                      ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "Results saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo "  - gazebo_ablation_results.json"
echo "  - gazebo_ablation_results.csv"
