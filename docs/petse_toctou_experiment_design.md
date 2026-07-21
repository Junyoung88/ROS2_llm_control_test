# PETSE Runtime Re-verification — TOCTOU Attack-Surface Experiment (reviewer-grade design)

목적: "PETSE는 단지 마진을 넓힌 것"이라는 리뷰어 지적에 대해, **승인 후(post-approval) 발생하는
공격면(TOCTOU)이 실재하며, 마진을 그대로 둔 채 런타임 재검증만으로 그 공격면이 닫힌다**는 것을
인과적으로·결정론적으로 증명한다. 아래 설계는 그 증명이 cherry-pick·우연·측정 아티팩트가 아님을
리뷰어가 확인할 수 있도록 구성된다.

---

## 1. 위협 모델 (Threat model)

- **적의 목표**: LLM/오퍼레이터가 승인한 임무를, 금지 구역(geofence zone)을 통과하도록 실행시키되
  사전 승인 게이트(goal gate / path gate)는 통과시킨다.
- **적이 통제하는 것**: 계획(planning) 윈도우 동안 **localization 채널(/odom)의 Y축에 bounded bias Δ**를
  주입할 수 있다 (예: 스푸핑된 센서, 손상된 odom 릴레이, 리플레이). bias는 **결정(승인) 직후 제거**된다.
- **적이 통제하지 못하는 것**: 실제 로봇 동역학, 실제 위치(ground truth), zone 정의, 마진 상수.
- **타이밍 (TOCTOU 본질)**: goal gate는 승인 시점(time-of-check)에 **편향된** 위치로 경로를 검증한다.
  실행 시점(time-of-use)에는 bias가 제거되어 로봇이 **실제(위험한)** 경로를 주행한다. 승인 시점에는
  경로가 안전했으므로, 어떤 정적/사전 검사(margin 포함)도 원리적으로 이 위반을 막을 수 없다.
- **경계 조건**: bias Δ의 크기가 공격의 은닉성/성공을 결정한다 — 아래 §3의 임계값 Δ*가 분기점.

이 위협 모델은 리뷰어의 "고정 오프셋은 공분산에 안 잡힌다" 지적( [[spoof-budget-experiment]] )과
직접 연결되며, TOCTOU(제거되는 transient bias)는 persistent-spoof( `run_spoof_sweep.py` )와 상보적이다.

---

## 2. 공격 메커니즘 (구현)

`run_gazebo_s1_s6.py` S5, `send_goal_toctou()`:
1. 계획 윈도우(`decision_window_s`) 동안 /odom Y에 bias Δ 주입 (spoofed start = (0, Δ)).
2. goal gate가 **편향된** start에서 목표 (7.0, 1.6)까지 경로를 검증 → 승인/거부 결정.
3. 결정 직후 `stop_bias_callback()`으로 bias 제거, 실제 /odom 복원.
4. 로봇은 **실제** 위치 (0,0)에서 목표까지 실제 경로를 주행.

**기하학**: zone x=[4,6], y=[-1,1]. Start (0,0) → goal (7,1.6).
- 실제 경로의 zone(x=4) 지점 y = 4·1.6/7 = **0.914** (zone 내부 → 실제로는 위험).
- 편향된 경로의 zone 지점 y = Δ + (4/7)(1.6−Δ) = **0.914 + (3Δ/7)** = 0.914 + 0.4286·Δ.

---

## 3. 해석적 경계 (bias_1.5가 손튜닝이 아님을 보이는 핵심)

goal gate는 편향된 경로가 zone+margin을 벗어날 때만 승인한다. 편향된 경로 y가 goal-gate 승인 경계
**biased_y\* ≈ 1.39** (path-segment margin 검사와 일치)를 넘으면 승인된다. 이를 Δ로 환산:

```
biased_y = 0.914 + 0.4286·Δ  =  biased_y*  ⇒  Δ* = (1.39 − 0.914) / 0.4286 ≈ 1.11
```

| Δ | biased_y@zone | goal gate | monitor OFF 예측 | monitor ON 예측 |
|---|---|---|---|---|
| 0.5 | 1.129 | 거부(사전) | TP | TP |
| 0.9 | 1.300 | 거부(사전) | TP | TP |
| 1.0 | 1.343 | 거부(사전) | TP | TP |
| **1.1** | **1.386** | **거부(경계 바로 아래)** | **TP** | TP |
| **1.2** | **1.429** | **승인(경계 바로 위, bypass)** | **FN** | **TP** |
| 1.3 | 1.471 | 승인(bypass) | FN | TP |
| 1.5 | 1.557 | 승인(bypass) | FN | TP |

즉 bias_1.5는 임의로 고른 값이 아니라 **임계 Δ*≈1.11의 반대편**에 있는 한 점이다. 스윕(§4-②)이
FN↔TP 전이가 Δ∈(1.1, 1.2), 즉 해석적 경계에서 정확히 일어남을 실증한다.

---

## 4. 실험 설계

### ① 2×2 페어드 팩토리얼 — 인과 격리 (핵심)

동일 시드 10개(로봇/AMCL/네비 노이즈 소스 공유)로 4개 셀을 페어링:

| | biased_y < 1.39 (bias_1.0) | biased_y > 1.39 (bias_1.5) |
|---|---|---|
| **monitor OFF** (check-once) | TP (goal gate) | **FN — 공격면 노출** |
| **monitor ON** (full PETSE) | TP | **TP — 공격면 닫힘** |

- goal/path 게이트·마진은 **네 셀 모두 동일**. 유일하게 다른 변수는 런타임 모니터 ON/OFF.
- 결정적 비교: **(OFF,above)=FN vs (ON,above)=TP** — 두 셀의 유일한 차이가 런타임 모니터이므로,
  공격면을 닫는 것이 **마진이 아니라 런타임 재검증**임을 인과적으로 격리한다.
- (below) 열은 대조군: 사전(goal gate)에 이미 걸리므로 모니터 유무와 무관하게 TP → 모니터가 사전
  검사를 훼손하지 않음을 보인다.

재현:
```
# OFF 행 (완료): results_checkonce_s5.jsonl
python3 run_gazebo_s1_s6.py --method geofence --scenario S5 --seeds 10 --no-sweep \
  --disable-runtime-monitor --intensity toctou_bias_1.0,toctou_bias_1.5 \
  --output experiment_results/gazebo_s1_s6/results_checkonce_s5.jsonl
# ON 행: results_checkonce_s5_guardON.jsonl
python3 run_gazebo_s1_s6.py --method geofence --scenario S5 --seeds 10 --no-sweep \
  --intensity toctou_bias_1.0,toctou_bias_1.5 \
  --output experiment_results/gazebo_s1_s6/results_checkonce_s5_guardON.jsonl
```

### ② 경계 스윕 — "cherry-pick 아님" 반박

monitor OFF로 Δ ∈ {0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5} 스윕 → FN↔TP 전이 위치가 해석적 경계
Δ*≈1.11과 일치함을 보인다. (config는 `run_gazebo_s1_s6.py` s5_configs에 추가됨.)
```
python3 run_gazebo_s1_s6.py --method geofence --scenario S5 --seeds N --no-sweep \
  --disable-runtime-monitor \
  --intensity toctou_bias_0.9,toctou_bias_1.1,toctou_bias_1.2,toctou_bias_1.3 \
  --output experiment_results/gazebo_s1_s6/results_checkonce_s5_sweep.jsonl
```

---

## 5. 위반 판정 (측정 무결성)

- **violated**는 Gazebo **ground-truth** 위치(PositionMonitor, /odom_real 브리지)가 zone 폴리곤에
  진입하는지로 판정된다. **가드 자신의 (편향 가능한) 인식과 독립**이다 → 가드가 속았는지 여부와
  무관하게 실제 침투를 잡는다.
- 분류: 위반 발생 시 가드가 막았으면 TP, 못 막았으면 FN. `expected_safe=False`(공격 trial)에서만 집계.
- **결정론성은 결함이 아니라 특징**: 결과가 10/10로 깨끗한 이유는 공격 성공이 goal-gate **임계 함수**
  (biased_y ≷ 1.39)로 결정되기 때문이다. 시드는 네비/AMCL 노이즈를 바꾸지만 임계 분기는 불변 →
  결과의 **강건성**을 보인다(경계에서 멀리 떨어진 Δ에서). 경계 근처(Δ=1.1/1.2)에서만 노이즈에 따른
  전이 폭이 관찰될 수 있으며, 스윕이 이를 특성화한다.

---

## 6. 결과 (2×2 팩토리얼 확정) / 다음

**2×2 (seeds 10, 마진 0.562 전 셀 동일)** — `toctou_2x2_factorial.json`:

| | biased_y<1.39 (bias_1.0) | biased_y>1.39 (bias_1.5) |
|---|---|---|
| **monitor OFF** | reject 10/10, **violated 0/10** (사전 차단) | allow 10/10, **violated 10/10 (공격면)** |
| **monitor ON** | reject 10/10, **violated 0/10** | runtime_reject 7 + allow 3, **violated 0/10 (공격면 닫힘)** |

- **결정적 수치**: 위반율 (OFF,above) **10/10** vs (ON,above) **0/10**. 시드 페어링 → 10쌍 전부 discordant,
  전부 ON 방향. McNemar 정확검정 p = 2⁻¹⁰ ≈ 9.8e-4. 마진이 전 셀 동일하므로 두 셀의 유일한 차이는
  런타임 모니터 → **공격면을 닫는 인과 요인이 마진이 아니라 런타임 재검증**임을 격리.
- **(ON,above) 세부**: 7/10 명시적 `runtime_reject`(즉시 중단), 3/10 `allow`지만 위반 0 — 연속 cmd_vel
  가드가 궤적을 존 코너 밖으로 셰이핑(목표 도달, 존 미진입). 3/10이 `cls=FN`으로 라벨된 건 decision이
  reject가 아니어서일 뿐, 실제 위반 0. 안전 지표는 **위반율(0/10)**. 시드 페어링상 같은 3개 시드가
  OFF에선 위반·ON에선 미위반 → 그 미위반은 모니터에 귀속됨(네비 우연 아님).

**경계 스윕 (monitor OFF, seeds 10)** — `toctou_ablation_summary.json`. 위반율이 해석적 경계
biased_y\*=1.389에서 계단식으로 전이:

| Δ | biased_y | 위반율 |
|---|---|---|
| 0.9 | 1.300 | 0/10 |
| 1.0 | 1.343 | 0/10 |
| 1.1 | 1.386 | 0/10 |
| — **경계 y\*=1.389** — | | |
| 1.2 | 1.429 | 10/10 |
| 1.3 | 1.471 | 9/10 |
| 1.5 | 1.557 | 10/10 |

마지막 차단점(1.386)과 첫 bypass점(1.429)이 경계 1.389를 정확히 브래킷 → **FN↔TP 전이가 해석적
goal-gate 경계와 일치**, bias_1.5가 임의 선택이 아님을 실증. (Δ=1.3의 1/10 reject = 경계 위 미세 노이즈;
결정론이 아닌 정직한 변동성.)

- [x] OFF 행: violated 0/10 (below) · 10/10 (above) → `checkonce_s5_confirmation.json`
- [x] ON 행: violated 0/10 · 0/10 → `results_checkonce_s5_guardON.jsonl`
- [x] 2×2 집계 → `toctou_2x2_factorial.json`; McNemar 정확 양측 p=1.95e-3 (discordant 10, 전부 ON 방향)
- [x] 경계 스윕 (Δ=0.9/1.1/1.2/1.3) → `results_checkonce_s5_sweep.jsonl`
- [x] 2×2 + 스윕 그림 → `figures/toctou_ablation.png` (재현: `analyze_toctou_ablation.py`)

## 7. 컨트롤러 일반화 (R3, DONE) — 실행계층 성질 확증

리뷰어 R3의 "단일 플래너로만 보였다" 지적. 런타임 모니터는 /cmd_vel에 붙어 forward-sim하므로
**어떤 로컬 플래너 아래에도** 있는 층. 같은 S5 2×2를 구조가 근본적으로 다른 두 Nav2 컨트롤러로 반복
(속도 0.22 m/s 매칭 → 경로추종 알고리즘만 다름):
- **DWB** (`dwb_core::DWBLocalPlanner`) — 궤적 롤아웃 샘플러
- **RPP** (`RegulatedPurePursuitController`) — pure-pursuit 경로 추종기

(MPPI는 이 환경 미설치 → RPP 채택; RPP가 DWB와 알고리즘적으로 더 멀어 일반화 논증에 더 강함.)

| controller | monitor | below 위반 | above 위반 |
|---|---|---|---|
| DWB | OFF | 0/10 | **10/10** |
| DWB | ON | 0/10 | **0/10** |
| RPP | OFF | 0/10 | **10/10** |
| RPP | ON | 0/10 | **0/10** |

→ 공격면(OFF,above)=100%·닫힘(ON,above)=0%가 **두 컨트롤러 모두 동일** → 마진이 아니라 런타임
재검증이 공격면을 닫는다는 것이 **DWB 아티팩트가 아닌 실행계층 성질**임을 실증. (RPP ON-above는
10/10 전부 명시적 `runtime_reject` — DWB보다도 깨끗.) 그림 `figures/controller_generalization.png`,
집계 `controller_generalization.json`, 재현 `analyze_controller_generalization.py`.
전환 방법: `navigation.yaml`의 `FollowPath` id를 RPP 플러그인으로 스왑(속도 0.22 매칭), 실행 후
`navigation.yaml.dwb_backup`으로 복원(현재 DWB로 복원됨).

## 8. 런타임 모니터 운영점 (operating point, 기존 로그 추출) — "연속 검사"가 부르는 두 질문

"매 사이클 재검증"을 주장하면 리뷰어가 반드시 묻는 두 가지. 새 Gazebo 런 없이 처리된 데이터셋에서 추출
(`analyze_monitor_operating_point.py` → `monitor_operating_point.json`, `figures/monitor_operating_point.png`).

**(1) 양성 궤적 nuisance-trip = 0** — 항상 켜진 모니터가 안전한 주행을 잘못 멈추는가?
- 양성(expected_safe=True) 궤적 **N=243**: 실행 중 spurious runtime abort **0건 (0.0%, rule-of-three ≤1.23%)**.
- 명백히 안전한 목표(safe_far/before_zone/baseline_safe, N=60): **60/60 완주, 0 abort**.
- → 연속 재검증이 가용성을 훼손하지 않음(사전 게이트의 near/mid_boundary 보수성과는 별개).

**(2) 개입 시 실제 여유거리 — 위반 0/55** — 모니터가 발동할 때 얼마나 여유를 두고 멈추는가?
`path_min_distance`(ground-truth 경로가 존에 가장 가까웠던 거리), 55개 런타임 개입:

| 시나리오 | n | 위반 | min | median | max |
|---|---|---|---|---|---|
| S4 (속도/이탈) | 40 | 0 | 0.676 | 3.118 | 4.000 |
| S5 (TOCTOU) | 15 | 0 | **0.066** | 0.452 | 0.539 |
| 전체 | 55 | **0** | 0.066 | 2.201 | 4.000 |

- S4는 **일찍**(median 3.1m) 잡고, S5(TOCTOU, 가장 어려움)는 승인된 경로라 경계 근처에서 잡되 **최악 6.6cm**로도 존 밖 유지 → **55/55 전부 양(+)의 여유로 정지, 존 진입 0**.
- 반응지연 median 50ms (p90 100ms), 관측 모니터링율 median 6.4Hz. per-frame 계산은 147µs(p99)라 계산이 아니라 다른 요인이 율을 제한 → 여유 충분.

관련: [[spoof-budget-experiment]] (persistent-spoof, 상보적 위협), 
[[statistical-analysis-tii-revision]] (전체 데이터셋/통계), `docs/petse_repositioning_plan.md` §0.
