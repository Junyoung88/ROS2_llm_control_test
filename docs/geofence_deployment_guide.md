# 지오펜스 시스템 배포 가이드

다른 ROS2 로봇에 `geofence_policy_enforcer` 패키지를 설치하고 설정하는 방법을 설명합니다.

---

## 1. 시스템 개요

지오펜스는 **2중 방어 구조**로 동작합니다:

```
[사용자/플래너]
    ↓ NavigateToPose
[goal_gate_node]  ← 목표 지점 검증 (planning-time)
    ↓ (승인 시)
[Nav2 Controller] → /cmd_vel_nav
    ↓
[cmd_vel_guard_node]  ← 속도 명령 필터 (runtime)
    ↓ (안전 시)
/cmd_vel → 로봇
```

| 노드 | 역할 |
|------|------|
| `goal_gate_node` | Nav2 목표 지점을 금지구역+안전 마진과 비교하여 승인/거부 |
| `cmd_vel_guard_node` | 속도 명령을 전방 시뮬레이션하여 위험 시 정지 |
| `path_watchdog_node` | 경로 모니터링 (선택사항) |
| `metrics_logger_node` | 메트릭 수집 (선택사항) |

### 안전 마진 공식

```
M(v) = z_{1-ε}·σ + (e₀ + c₁·v) + v·τ + v²/(2·a_max)
       ─────────   ────────────   ─────   ───────────
       측위 불확실성  경로 추종 오차   지연 보상  제동 거리
```

---

## 2. 요구사항

### ROS2 패키지
- ROS2 Jazzy (또는 Humble)
- `nav2_msgs` — NavigateToPose, FollowWaypoints 액션
- `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`
- `visualization_msgs` — RViz 시각화
- `tf2_ros`, `tf2_geometry_msgs` — TF 변환

### Python 패키지
- `numpy` — 필수
- `pyyaml` — 필수 (설정 파일 파싱)
- `shapely` — 선택 (없으면 내장 ray-casting으로 대체)

### Nav2 토픽 요구사항
로봇에 다음 토픽이 존재해야 합니다:

| 토픽 | 타입 | 용도 |
|------|------|------|
| `/odom` | nav_msgs/Odometry | 로봇 위치·속도 |
| `/cmd_vel_nav` | geometry_msgs/Twist | Nav2 컨트롤러 출력 (guard 입력) |
| `/cmd_vel` | geometry_msgs/Twist | 최종 속도 명령 (로봇 입력) |
| `navigate_to_pose` | nav2_msgs action | Nav2 네비게이션 액션 |

---

## 3. 패키지 설치

### 3.1 소스 복사

```bash
# 대상 워크스페이스에 패키지 복사
cp -r geofence_policy_enforcer/ ~/your_robot_ws/src/
```

복사할 파일 구조:
```
geofence_policy_enforcer/
├── geofence_policy_enforcer/   # Python 소스
│   ├── __init__.py
│   ├── geofence_core.py        # 핵심: 구역 판정 + 마진 계산
│   ├── cmd_vel_guard_node.py   # 핵심: 속도 필터 노드
│   ├── goal_gate_node.py       # 핵심: 목표 검증 노드
│   ├── path_watchdog_node.py   # 선택: 경로 감시
│   └── metrics_logger_node.py  # 선택: 메트릭 수집
├── launch/
│   └── demo.launch.py
├── config/
│   ├── geofence.yaml           # ← 로봇별 수정 필요
│   └── geofence_params.yaml
├── resource/
│   └── geofence_policy_enforcer
├── package.xml
└── setup.py
```

> **참고**: `attack_*.py`, `safety_baselines.py`, `llm_command_parser.py` 등은 실험용이므로 배포 시 제외 가능합니다.

### 3.2 빌드

```bash
cd ~/your_robot_ws
colcon build --packages-select geofence_policy_enforcer
source install/setup.bash
```

---

## 4. 로봇별 설정

### 4.1 금지구역 설정 (`config/geofence.yaml`)

로봇 운용 환경에 맞게 금지구역을 정의합니다:

```yaml
uncertainty:
  localization_sigma: 0.15    # 측위 표준편차 (m) — 로봇 AMCL/EKF에 맞게 조정
  v_max: 0.5                  # 최대 속도 (m/s) — Nav2 설정과 일치
  latency: 0.1                # 시스템 지연 (s) — 네트워크/제어 루프 지연
  a_max: 2.5                  # 최대 감속 (m/s²) — 로봇 물리 한계
  k_sigma: 3.0                # legacy (epsilon 사용 시 무시)

zones:
  # 금지구역: 로봇이 절대 진입하면 안 되는 영역
  - name: "server_room"
    type: "forbidden"
    priority: 10
    vertices:
      - {x: 5.0, y: 2.0}
      - {x: 8.0, y: 2.0}
      - {x: 8.0, y: 5.0}
      - {x: 5.0, y: 5.0}

  # 금지구역을 여러 개 정의 가능
  - name: "elevator_area"
    type: "forbidden"
    priority: 10
    vertices:
      - {x: 12.0, y: -1.0}
      - {x: 14.0, y: -1.0}
      - {x: 14.0, y: 1.0}
      - {x: 12.0, y: 1.0}

  # 안전 구역 (시각화용, 선택)
  - name: "operating_area"
    type: "safe"
    priority: 0
    vertices:
      - {x: -10.0, y: -10.0}
      - {x: 20.0, y: -10.0}
      - {x: 20.0, y: 10.0}
      - {x: -10.0, y: 10.0}
```

**구역 타입**:
- `forbidden` — 금지구역. 마진 포함하여 거부/차단
- `buffer` — 경고구역. 진입 시 경고만 발행
- `safe` — 안전구역 (시각화 전용)

**꼭짓점 좌표**: 맵 프레임(`map`) 기준 (x, y). 최소 3개 (삼각형). 시계/반시계 무관.

### 4.2 로봇 파라미터 측정 및 설정

배포 전에 아래 값들을 로봇에서 실측해야 합니다:

| 파라미터 | 측정 방법 | 기본값 |
|---------|----------|-------|
| `localization_sigma` | AMCL covariance 또는 EKF 로그에서 위치 표준편차 확인 | 0.15 m |
| `v_max` | Nav2 controller의 `max_vel_x` 파라미터 확인 | 0.5 m/s |
| `a_max` | Nav2 controller의 `decel_lim_x` 절대값 확인 | 2.5 m/s² |
| `latency` | cmd_vel 발행 → odom 반영 시간차 측정 (rosbag 분석) | 0.1 s |
| `epsilon` | 허용 위험 수준 (0.003 = 99.7% 안전) | 0.003 |

**측위 불확실성 확인 예시:**
```bash
# AMCL covariance 확인
ros2 topic echo /amcl_pose --field pose.covariance --once
# → [0]과 [7] 값의 sqrt가 대략적인 sigma
```

**제동 능력 확인 예시:**
```bash
# DWB local planner의 감속 한계
ros2 param get /controller_server dwb_core.decel_lim_x
# → -2.5 (부호 무시, 절대값 사용)
```

### 4.3 토픽 매핑

로봇의 토픽 이름이 기본값과 다를 경우, launch 인자로 변경합니다:

| 기본 토픽 | launch 인자 | 설명 |
|-----------|------------|------|
| `/cmd_vel_nav` | `cmd_vel_input_topic` | guard 입력 (Nav2 출력) |
| `/cmd_vel` | `cmd_vel_output_topic` | guard 출력 (로봇 입력) |

**토픽 체인 구성:**

Nav2가 `/cmd_vel`로 출력하는 경우 (기본):
```
Nav2 → /cmd_vel → (remapping 필요) → /cmd_vel_nav → guard → /cmd_vel → 로봇
```

Nav2 controller의 output 토픽을 `/cmd_vel_nav`로 remap하거나, launch 인자로 guard의 입출력을 조정하세요.

---

## 5. 실행

### 5.1 launch 파일로 실행

```bash
# 기본 실행 (goal_gate + cmd_vel_guard 모두 활성화)
ros2 launch geofence_policy_enforcer demo.launch.py \
  geofence_config:=/path/to/your/geofence.yaml \
  use_sim_time:=false

# 실제 로봇 (토픽 커스텀)
ros2 launch geofence_policy_enforcer demo.launch.py \
  geofence_config:=/path/to/your/geofence.yaml \
  use_sim_time:=false \
  cmd_vel_input_topic:=/cmd_vel_nav \
  cmd_vel_output_topic:=/cmd_vel \
  v_max:=0.3 \
  a_max:=1.5 \
  localization_sigma:=0.10 \
  latency:=0.05

# goal_gate만 사용 (runtime guard 불필요 시)
ros2 launch geofence_policy_enforcer demo.launch.py \
  geofence_config:=/path/to/your/geofence.yaml \
  enable_cmd_vel_guard:=false \
  enable_path_watchdog:=false \
  enable_metrics:=false
```

### 5.2 개별 노드 실행

launch 파일 대신 노드를 직접 실행할 수도 있습니다:

```bash
# Goal Gate만 실행
ros2 run geofence_policy_enforcer goal_gate_node \
  --ros-args \
  --params-file /path/to/geofence_params.yaml \
  -p geofence_config:=/path/to/geofence.yaml \
  -p use_sim_time:=false \
  -p safety_method:=geofence \
  -p v_max:=0.5 \
  -p latency:=0.1

# Cmd Vel Guard만 실행
ros2 run geofence_policy_enforcer cmd_vel_guard_node \
  --ros-args \
  --params-file /path/to/geofence_params.yaml \
  -p geofence_config:=/path/to/geofence.yaml \
  -p use_sim_time:=false \
  -p input_topic:=/cmd_vel_nav \
  -p output_topic:=/cmd_vel
```

> **중요**: `--params-file`은 반드시 `-p` 옵션보다 **앞에** 와야 합니다. ROS2에서는 뒤에 오는 인자가 우선하기 때문입니다.

### 5.3 Nav2 launch에 통합

기존 Nav2 launch 파일에 지오펜스를 포함시키려면:

```python
# your_robot_bringup.launch.py
from launch_ros.actions import Node

goal_gate = Node(
    package='geofence_policy_enforcer',
    executable='goal_gate_node',
    name='goal_gate',
    parameters=[{
        'use_sim_time': False,
        'geofence_config': '/path/to/geofence.yaml',
        'safety_method': 'geofence',
        'v_max': 0.5,
        'latency': 0.1,
        'a_max': 2.5,
        'localization_sigma': 0.15,
        'use_epsilon_formulation': True,
        'epsilon': 0.003,
    }],
)

cmd_vel_guard = Node(
    package='geofence_policy_enforcer',
    executable='cmd_vel_guard_node',
    name='cmd_vel_guard',
    parameters=[{
        'use_sim_time': False,
        'geofence_config': '/path/to/geofence.yaml',
        'input_topic': '/cmd_vel_nav',
        'output_topic': '/cmd_vel',
        'safety_method': 'geofence',
        'override_strategy': 'stop',
    }],
)
```

---

## 6. 동작 확인

### 6.1 노드 상태 확인

```bash
# 노드 목록에서 확인
ros2 node list | grep geofence

# goal_gate 상태
ros2 topic echo /geofence/status --once

# cmd_vel_guard 상태
ros2 topic echo /geofence/cmd_vel_status --once
```

### 6.2 구역 시각화 (RViz)

RViz에서 `/geofence/zones_viz` (MarkerArray)를 추가하면 금지구역이 빨간색 다각형으로 표시됩니다.

### 6.3 목표 지점 테스트

```bash
# 금지구역 내부로 목표 전송 → 거부되어야 함
ros2 action send_goal /navigate_to_pose_safe nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 5.5, y: 0.0}}}}"

# 이벤트 로그 확인
ros2 topic echo /geofence/goal_events
```

### 6.4 마진 분해 확인

```bash
# 현재 마진 각 항목 확인
ros2 topic echo /geofence/margin_breakdown --once
```

출력 예시:
```json
{
  "total_margin": 0.562,
  "estimation_term": 0.412,
  "tracking_term": 0.050,
  "latency_term": 0.050,
  "braking_term": 0.050
}
```

### 6.5 런타임 설정 변경

```bash
# 설정 파일 리로드
ros2 service call /geofence/reload std_srvs/srv/Trigger
```

---

## 7. 토픽 구성도

```
구독 (Subscriptions)
├── /odom                  [Odometry]       로봇 위치·속도
├── /cmd_vel_nav           [Twist]          Nav2 출력 (guard 입력)
├── /plan                  [Path]           계획 경로 (tracking error 측정)
└── /geofence/nav_state    [String]         네비게이션 승인 상태

발행 (Publications)
├── /cmd_vel               [Twist]          필터링된 속도 명령
├── /geofence/goal_events  [String]         목표 판정 이벤트 (JSON)
├── /geofence/zones_viz    [MarkerArray]    구역 시각화
├── /geofence/status       [String]         노드 상태 (JSON)
├── /geofence/nav_state    [String]         네비게이션 상태
└── /geofence/cmd_vel_events [String]       속도 필터 이벤트 (JSON)

액션 서버 (goal_gate)
├── /navigate_to_pose_safe     ← 여기로 목표 전송
└── /follow_waypoints_safe     ← 여기로 웨이포인트 전송

액션 클라이언트 (goal_gate → Nav2)
├── /navigate_to_pose          → Nav2로 전달
└── /follow_waypoints          → Nav2로 전달
```

---

## 8. 주요 파라미터 요약

### 마진 관련 (가장 중요)

| 파라미터 | 설명 | 기본값 | 조정 기준 |
|---------|------|-------|----------|
| `epsilon` | 허용 위험률 | 0.003 | 낮출수록 마진 증가 (더 보수적) |
| `localization_sigma` | 측위 σ | 0.15 m | AMCL/EKF covariance에서 측정 |
| `v_max` | 최대 속도 | 0.5 m/s | Nav2 max_vel_x와 동일하게 |
| `a_max` | 최대 감속 | 2.5 m/s² | Nav2 decel_lim_x 절대값 |
| `latency` | 시스템 지연 | 0.1 s | cmd_vel→odom 시간차 측정 |
| `e_0` | 정적 추종 오차 | 0.03 m | 경로 추종 정밀도 측정 |
| `c_1` | 속도 비례 추종 계수 | 0.04 | 고속 주행 시 추종 오차 측정 |

### 동작 모드

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| `safety_method` | 안전 방법 | `geofence` |
| `enable_goal_gate` | 목표 검증 활성화 | `true` |
| `enable_cmd_vel_guard` | 속도 필터 활성화 | `true` |
| `override_strategy` | 위반 시 대응 | `stop` |
| `use_epsilon_formulation` | ε 기반 마진 사용 | `true` |
| `use_sim_time` | 시뮬레이션 시간 사용 | `true` |

---

## 9. 트러블슈팅

### guard가 모든 명령을 차단함
- `localization_sigma`나 `latency`가 너무 크면 마진이 과도하게 커집니다
- `ros2 topic echo /geofence/margin_breakdown`으로 마진 각 항목 확인
- `epsilon`을 높이면 (예: 0.01) 마진이 줄어듦

### goal_gate가 목표를 거부하지 않음
- `geofence_config` 경로가 올바른지 확인
- `ros2 topic echo /geofence/status`에서 구역이 로드되었는지 확인
- `safety_method`가 `geofence`인지 확인 (`no_guard`면 모두 통과)

### cmd_vel_guard가 동작하지 않음
- 입력 토픽(`cmd_vel_input_topic`)에 실제로 데이터가 오는지 확인: `ros2 topic hz /cmd_vel_nav`
- `/odom`에 데이터가 오는지 확인: `ros2 topic hz /odom`
- standalone으로 실행 중인지 확인 (launch file 내 DDS 격리 문제 가능)

### DDS 관련 문제
- `cmd_vel_guard_node`가 launch file 안에서 구독 데이터를 못 받는 경우, standalone(`ros2 run`)으로 실행
- CycloneDDS 사용 시 SIGKILL 후 좀비 participant가 남을 수 있음 → 프로세스 완전 재시작

---

## 10. 최소 설치 (goal_gate만)

runtime 속도 필터가 필요 없고, 목표 지점 검증만 원하는 경우:

```
필요 파일:
  geofence_core.py
  goal_gate_node.py
  __init__.py
  config/geofence.yaml
  package.xml, setup.py (entry_points에서 goal_gate_node만 유지)
```

```bash
ros2 run geofence_policy_enforcer goal_gate_node \
  --ros-args \
  -p geofence_config:=/path/to/geofence.yaml \
  -p use_sim_time:=false \
  -p safety_method:=geofence
```

사용자는 `/navigate_to_pose` 대신 `/navigate_to_pose_safe`로 목표를 보내면 됩니다.
