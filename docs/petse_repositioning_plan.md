# PETSE 재포지셔닝 계획 (① 포지셔닝 + COE/TCB + 주장 수정)

---

## ⭐ 0. 논문의 진짜 기여 재정의 (마진 → 연속 재검증)

**문제의식**: 리뷰어는 PETSE를 "마진만 넓힌 것"으로 읽었다. 실제 기여는 마진이 아니라 **승인된 작업을 실행 중 연속 재검증(runtime re-verification)하는 것** — LLM/planner가 승인해도 그 승인은 취소 가능하며, 실행 중 물리 상태가 검사 시점과 달라지면 철회한다. TOCTOU(Time-Of-Check ≠ Time-Of-Use) 개념.

**결정적 근거 — check-once vs continuous ablation** (`run_checkonce_ablation.py`):
마진을 **0.562m로 고정**한 채 런타임 모니터만 제거(=승인 시 1회 검사)한 반사실:
| 시나리오(논문) | full recall | check-once(마진 동일) | 런타임만 잡음 |
|---|---|---|---|
| S1 (goal) | 100% | 100% | 0 |
| S2 (path) | 100% | 100% | 0 |
| **S3 (velocity/deviate)** | 100% | **0%** | 40 |
| **S4 (spoof/TOCTOU)** | 100% | 75% | 15 |
| **전체 (n=200)** | 100% | **72.5%** | 55 |

→ **마진이 똑같아도 런타임 모니터를 끄면 recall 100%→72.5%.** S3(승인 후 속도조작/이탈)는 **0%** — 마진이 아무리 커도 승인 시점엔 안전했으므로 static check로는 원리적으로 못 잡는다. 잃은 55건은 전부 **승인 후 발생한 문제**. 그림 `figures/checkonce_ablation.png`.

**함의 (논문 전면 재구성)**:
- **기여의 주인공 = 런타임 모니터(연속 재검증)**, 마진은 "모니터가 매 사이클 강제하는 조건"으로 격하.
- 기존 planning-layer 방어(SELP/SafetyChip/shielding)는 **승인 시 1회 검사** → S3/S4를 원리적으로 못 잡음. 이게 계층 차별화의 본질(마진 크기 아님).
- CBF 민감도(§J)와 결합: CBF도 goal-only+inside-loop이라 S2/S3 구조적 실패 → "check-once의 한계"를 다른 각도에서 재확인.
- **"마진만 넓힌 것 아니냐"(R1.1/R1.2/AE) 근본 무력화**: 마진 고정 실험이 마진과 무관하게 연속 재검증이 방어의 27.5%p를 담당함을 증명.

**논문 반영**: 이 표/그림을 **메인 결과의 첫 번째**로. 기여문(§B) ①을 "risk-parameterized margin"에서 **"revocable approval via continuous runtime re-verification"**으로 승격, 마진은 ②로 강등. Ablation 섹션의 주력을 마진 10x가 아니라 이 check-once ablation으로 교체. 제목/초록도 "runtime re-verification"을 전면에.

**Gazebo 실물 확증 (V-check-once)**: 위 표는 full 데이터의 게이트별 반사실이므로, 실제로 런타임 모니터를 끄고 재실행해 확증. 러너에 `--disable-runtime-monitor` 플래그 추가(geofence 가드 강제 OFF, goal/path 게이트+마진 불변). S4 geofence 6 trials(`results_checkonce.jsonl`) 결과:
- 공격이 실제로 존에 도달한 trial 2개(path_min=0.00m): **둘 다 존 진입**(위반 12·22회). full-PETSE 가드 ON에선 S4가 41/41 TP·**0 위반**.
- 나머지 4개: 공격이 존 미도달(path_min 2.0–3.0m) — S4 direct_control 공격의 알려진 ~33% 성공률(불안정). 방어 미테스트.
- **결론**: 마진 그대로여도 런타임 모니터를 끄면, 도달한 공격은 **전부** 존에 진입 → 반사실의 물리적 확증. (버그 수정: S4 분류의 `s4_guard_active`가 method 하드코딩 → `disable_runtime_monitor` 반영하도록 수정.)
- 재현: `python3 run_gazebo_s1_s6.py --method geofence --scenario S4 --seeds 3 --disable-runtime-monitor --output results_checkonce.jsonl`. 더 깨끗한 aggregate 원하면 공격 신뢰도 개선 후 시드 확대 필요.

**⭐ 큰 표본 확증 (S5 TOCTOU, 결정론적)**: S4 direct_control은 ~33% 성공률로 불안정 → S5 TOCTOU(결정론적 goal-gate 탐지)로 전환해 시드 확대. 러너: `--method geofence --scenario S5 --seeds 10 --disable-runtime-monitor --intensity toctou_bias_1.0,toctou_bias_1.5 --output results_checkonce_s5.jsonl`. 마진 0.562m 고정, goal-gate 경계 y=1.389. 결과(`checkonce_s5_confirmation.json`), 완벽하게 깨끗:

| intensity | biased y | vs 경계 1.389 | 가드 OFF 결과 | n |
|---|---|---|---|---|
| `toctou_bias_1.0` | 1.343 | 아래 → goal gate가 거부 | 10/10 **reject → TP** | 10 |
| `toctou_bias_1.5` | 1.557 | 위 → goal gate 기만 | 10/10 **allow → violated → FN** | 10 |

→ bias_1.0은 승인 시점(pre-motion)에 이미 걸리므로 런타임 모니터를 꺼도 check-once가 유지(10/10 TP). bias_1.5는 goal gate를 통과 → 런타임 모니터가 없으면 로봇이 실제 경로(y=0.914)로 존에 진입 → **10/10 위반·FN**. Full PETSE(런타임 모니터 ON)에선 이 경우가 런타임 TOCTOU 가드에 걸려 TP. **마진 불변 상태에서 런타임-only 탐지가 전부 FN으로 뒤집힘을 결정론적·물리적으로 확증** — check-once ablation의 반사실을 시드 10개로 재확인.

**⭐⭐ 2×2 페어드 팩토리얼 (인과 격리, `toctou_2x2_factorial.json`)**: 위 OFF 행에 **guard-ON 행을 같은 시드**로 추가해 2×2 완성 (마진 0.562 전 셀 동일):

| | biased_y<1.39 (bias_1.0) | biased_y>1.39 (bias_1.5) |
|---|---|---|
| **monitor OFF** | 위반 0/10 (사전 차단) | **위반 10/10 (공격면)** |
| **monitor ON** | 위반 0/10 | **위반 0/10 (공격면 닫힘)** |

**핵심**: 위반율 (OFF,above) 10/10 vs (ON,above) 0/10. 시드 페어링 → 10쌍 전부 discordant·전부 ON 방향, McNemar 정확 양측 p=1.95e-3. 전 셀 마진 동일 → 두 셀 유일 차이는 런타임 모니터 → **공격면을 닫는 것은 마진이 아니라 런타임 재검증**임을 인과적으로 격리(리뷰어의 "그냥 마진" 지적 정면 반박). (ON,above): 7 명시적 runtime_reject + 3 연속 cmd_vel 셰이핑(존 미진입); 3개 cls=FN은 decision 라벨링 아티팩트, 실제 위반 0. 상세: `docs/petse_toctou_experiment_design.md`(위협모델·해석적 경계 Δ*≈1.11·측정 무결성).

**경계 스윕 (monitor OFF, `toctou_ablation_summary.json`)**: Δ=0.9/1.0/1.1/1.2/1.3/1.5 → biased_y 1.300~1.557, 위반율 0/0/0/10/9/10 (÷10). 마지막 차단(1.386)·첫 bypass(1.429)가 해석적 경계 1.389를 정확히 브래킷 → **FN↔TP 전이가 예측 경계와 일치**, bias_1.5가 cherry-pick 아님 실증. **그림**: `figures/toctou_ablation.png` (2×2 히트맵 + 스윕 계단; 재현 `analyze_toctou_ablation.py`) → 논문 메인 결과 그림.

---


대상: `conference_101719_8p.tex` (2026-04-07 버전, TII 투고본) → RA-L/RAS 재투고본.
원칙: **주장을 좁혀서 완전히 입증 가능하게 만든다.** 실험 추가 없이 글로 해결되는 것만 여기서 다룬다.

---

## A. 포지셔닝 전략

현재 논문은 세 정체성이 혼재: ① LLM 로봇 보안, ② 불확실성 인지 지오펜싱, ③ 실행계층 안전. 리뷰어 1·AE의 "독창성 부족" 공격은 ①을 전면에 세운 대가다 (LLM 부분이 방법론에 없으니 "동기 수준"이라는 지적이 성립).

**새 중심 주장:**

> PETSE는 LLM을 보호하거나 악성 프롬프트를 탐지하는 기법이 아니다. 상류 명령 생성·계획 계층이 신뢰될 수 없을 때(LLM 명령 인터페이스가 대표 사례), 실행 시점에 공간 안전 제약을 강제하는 **runtime assurance layer**이며, 기여의 핵심은 **운영자가 지정한 위반 확률 ε로부터 유도되는 실행계층 동적 마진 공식과 그 체계적 검증**이다.

LLM의 새 위치: (a) 공격 입력 생성 장벽을 낮추는 upstream command interface, (b) PETSE가 보호하는 여러 명령 소스 중 하나, (c) 방법론의 필수 구성요소 아님 — Introduction 1–2문단과 S1/S2(salami) 시나리오 동기로만 사용.

**차별화 논거(독창성 방어) 3줄 요약** — Related Work 끝과 Introduction에 명시:
1. 기존 실행계층 모니터(정적 지오펜스, ROS 2 Collision Monitor, RoboGuard류)는 마진이 고정이거나 경험적이다. PETSE는 마진을 **측정 가능한 시스템 파라미터 + 운영자 위험 허용치 ε**에서 유도한다 (ε → z-quantile → margin, 검증된 risk knob).
2. goal gate + path gate + runtime monitor의 결합은 goal-only(S1), path-through(S2/S3), 승인 후 일탈·TOCTOU(S4/S5)를 **한 계층에서** 커버한다 — 어느 단일 기존 기법도 전부 커버하지 못함을 베이스라인 실험이 보여준다.
3. 동일 마진을 CBF에 이식한 CBF-Adaptive가 여전히 실패하는 실험(격리 실험)이 "마진 크기가 아니라 구조가 기여"임을 입증한다.

**제목 후보** (LLM은 부제로 격하):
- 1안(추천): *PETSE: An Uncertainty-Aware Runtime Safety Envelope for Mobile Robots under Compromised Navigation Commands*
- 2안: *PETSE: Probabilistic Runtime Enforcement of Spatial Safety for Mobile Robots with Untrusted Command Pipelines*
- 3안(LLM 유지): *PETSE: Runtime Spatial Safety Enforcement for LLM-Enabled Mobile Robots under Execution Uncertainty*

---

## B. Contributions 재작성 (tex L65–77 교체)

```latex
\begin{enumerate}
\item \textbf{Risk-parameterized execution-layer margin.}
A dynamic safety margin derived from four measurable system
parameters---pose-covariance radius, cross-track envelope,
latency displacement, and braking distance---whose estimation
term is calibrated by an operator-chosen violation tolerance
$\varepsilon$, providing an explicit safety--availability knob.

\item \textbf{Scoped runtime assurance guarantee.}
Within an explicitly stated Certified Operating Envelope (COE),
forbidden-zone avoidance is guaranteed independently of the
correctness of all upstream command-generation and planning
components; outside the COE, violations of the operating
assumptions are detected and answered with a fail-stop.

\item \textbf{Three-gate enforcement architecture.}
A goal gate, a path gate, and a per-cycle runtime monitor,
implemented as a single ROS~2 lifecycle node that drops into an
unmodified Nav2 stack.

\item \textbf{Systematic adversarial evaluation.}
Four attack scenarios spanning goal injection, path inducement,
runtime deviation, and localization spoofing, evaluated over
N Gazebo trials (20 random seeds) against six baselines including
two execution-layer monitors, with confidence intervals and
paired significance tests, plus hardware validation on a second
platform.
\end{enumerate}
```

주의: 기여 4의 시나리오 목록·수치는 새 실험 테이블 확정 후 채움. **기존 기여 1의 "survives a fully compromised command pipeline"은 폐기** (E-1 참조).

---

## C. 신규 섹션 초안 — Threat Model 교체 (tex §III)

### C-1. Trusted Computing Base 표 (신규 Table)

```latex
\begin{table}[t]
\centering\footnotesize
\caption{Trust assumptions and defense scope.}
\label{tab:tcb}
\begin{tabular}{@{}l l l@{}}
\toprule
Component & Trust & PETSE defends? \\
\midrule
LLM / NL command interface   & untrusted        & yes (S1, S2) \\
Navigation goals \& plans    & untrusted        & yes (S1--S3) \\
Local controller output      & untrusted        & yes (S4) \\
Pose estimate (AMCL/odom)    & bounded-spoof$^{*}$ & conditional (S5) \\
Reported covariance          & partially trusted & detection only \\
PETSE node \& stop channel   & trusted (TCB)    & out of scope \\
Motor driver / firmware      & trusted (TCB)    & out of scope \\
\bottomrule
\multicolumn{3}{@{}p{0.95\columnwidth}}{\footnotesize $^{*}$Spoofing
offsets bounded by $\Delta_{\mathrm{spoof}}$; arbitrary stealthy
spoofing with consistent covariance is outside the guarantee
(Section~Limitations).}
\end{tabular}
\end{table}
```

본문에 명시할 문장:

> PETSE does not guarantee safety if both the navigation stack and the independent stop-enforcement path are compromised. The PETSE node, its zero-velocity channel, and the motor driver form the trusted computing base; industrial deployments realize this via a safety PLC or a read-only-firmware compute module, and hardening our prototype to that level is engineering future work.

### C-2. Certified Operating Envelope (신규 Definition, §IV Theorem 1 앞)

```latex
\begin{definition}[Certified Operating Envelope]\label{def:coe}
The COE is the operating region in which
$\|v(t)\| \le v_{\max}$, $\tau(t) \le \tau_{\max}$,
$a_{\mathrm{dec}}(t) \ge a_{\min}$,
$\lambda_{\max}(\Sigma_t) \le \bar\lambda$, and the tracking error
lies within the fitted envelope of Eq.~(3). Theorem~1 certifies
forbidden-zone avoidance \emph{inside} the COE. Outside it, PETSE
provides detection and fail-stop, not avoidance certification.
\end{definition}
```

이어서 안전 주장 3단계를 한 문단으로:

> PETSE therefore makes three claims of decreasing strength: (i) *certified avoidance* inside the COE (Theorem 1); (ii) *violation detection* of every COE precondition, each of which is monitored online ($\|v_t\|$, $\lambda_{\max}(\Sigma_t)$, sensor timeouts, latency threshold); and (iii) *fail-stop response* upon detection. We do not claim safety when an attacker both breaks a COE precondition and defeats the fail-stop path — that case requires the trusted stop channel of Table~\ref{tab:tcb}.

이 구분은 리뷰어 1의 "가정이 깨지면 보장 못 한다"와 "fail-stop은 단순 비상수단"을 동시에 처리한다: 우리는 그 경우 avoidance를 주장한 적 없고, detection+fail-stop이 그 영역의 설계된 응답임을 명시.

### C-3. 스푸핑 위협의 한정 (S5 서술과 Limitations) — ②단계 실험으로 뒷받침됨

**실험**: `run_spoof_sweep.py` — 지속적(stealthy) pose 스푸핑 magnitude sweep. 오프셋 δ를 전 구간 유지(TOCTOU처럼 제거하지 않음)하고 runtime monitor가 스푸핑된 위치를 신뢰할 때 실제 구역 침투를 측정. 결과 `spoof_sweep_results.json`, 그림 `figures/spoof_budget_sweep.png`.

**핵심 결과 (정확한 임계값)**:
$$\text{penetration} = \delta - (M_{\mathrm{est}} + M_{\mathrm{track}}), \quad \Delta_{\mathrm{spoof}} = M_{\mathrm{est}} + M_{\mathrm{track}} = 0.462\,\text{m}$$
- 마진의 **공간항(추정+추종)만** 지속-스푸핑 예산을 제공한다. 지연·제동항은 실제 주행에 소비되어 스푸핑 여유가 없다 — 논문의 "covariance-based margin absorbed the offset"를 정밀화한 주장.
- **Gazebo 실측(3 seeds, 27 trials, std≤0.01m)**: 침투가 예측을 일정 오프셋(+0.055m, 실제 로봇 가속 램프/제동 오버슈트)으로 추종. **경험적 임계값 ≈0.403m** vs 인증 예산 0.462m — 실제 동역학이 약간 일찍 위반시키므로 인증값이 낙관적임을 정직하게 보고 (안전 방향으로는 보수적으로 재조정 근거).
- δ ≤ 0.40m → 흡수(안전), δ ≥ 0.462m → 3/3 위반. 그림 `figures/spoof_budget_sweep.png`.
- **실측이 인증값보다 작다는 점**은 마진에 실제-동역학 여유(예: M_track 상향 또는 별도 여유항)를 두어야 함을 시사 — Limitations/future work에 명시.

**논문 반영**:
- 위협 모델에 $\|\delta_{\mathrm{spoof}}(t)\| \le \Delta_{\mathrm{spoof}} = M_{\mathrm{est}}+M_{\mathrm{track}}$ 명시. 실기체 S4의 0.5m 오프셋이 왜 막혔는지 정량 설명(0.5 < 0.462? — 실기체는 σ/τ가 달라 M_est가 큼; 실기체 M≈0.47이므로 재계산 필요, E-7과 연동).
- 하드웨어 S4 오프셋(0.5m)이 시뮬 예산(0.462m)에 근접함을 명시하고, 실기체 파라미터 기준 예산으로 재서술.
- "covariance-based margin absorbed the offset" → E-7 문구로 교체.
- Limitations: stealthy spoofing(공분산 작게 유지 + δ>Δ_spoof)은 보장 밖. 탐지책으로 odom–AMCL residual 검사 $r_t = \|\Delta p_{\mathrm{AMCL}} - \Delta p_{\mathrm{odom}}\| > \gamma$를 future work로 명시(구현은 하지 않되 설계 제시).
- **Stealthy vs detectable 구분**: 공분산을 부풀리는 스푸핑은 fail-stop(λ_max>λ̄)으로 막힘 → 마진 무관하게 완화. 공격자는 공분산을 작게 유지(stealthy)해야 하며, 그때만 Δ_spoof 예산이 유일한 방어선. 이 구분이 리뷰어의 "fixed offset은 covariance에 안 나타난다" 지적에 대한 정확한 답.

---

## D. Related Work 보강 (리뷰어 3: "기존 실행계층 방어 미인용")

추가 조사·인용할 카테고리 (RA-L 전 필수):
1. **Runtime assurance / Simplex architecture** — Sha (2001) "Using Simplicity to Control Complexity"; RTA 서베이 (Schierman et al.). PETSE = geometric RTA monitor로 위치.
2. **ROS 2 Nav2 Collision Monitor / velocity smoother** — 실배포 실행계층 안전장치. 왜 부족한가: 장애물(센서 기반)용이며 금지구역·불확실성·지연 비인지.
3. **RoboGuard 류 LLM-로봇 런타임 가드** — 이미 베이스라인으로 구현되어 있음(METHOD_ORDER에 존재). Related Work에 인용 + 실험에서 비교됨을 명시.
4. **Reachability 기반 온라인 안전 모니터** — Althoff 계열 online reachset monitoring.
5. **UAV 동적 지오펜싱** — 기존 인용(stevens, vagal) 확장.

수정할 문장:
- L130 "No prior method that we are aware of derives..." → 유지 가능하되 대상을 좁혀서: "derives an execution-layer margin **from an operator-specified violation tolerance ε jointly with** covariance, tracking, latency, and braking bounds".
- L544 Limitations "a peer execution-layer defense---one that, to our knowledge, does not yet exist" → **삭제 필수.** 새 실험에 RoboGuard + Static Geofence(CBF 0.3m가 사실상 static 변형) + CBF-Adaptive가 있으므로: "We compare against two execution-layer monitors (a static geofence and RoboGuard) and an adaptive-margin variant (CBF-Adaptive); a fully independent adaptive geometric monitor from the literature does not exist to our knowledge, which is why CBF-Adaptive serves as the closest same-margin control."
- Table I (L109): "Attack Robust" 열 이름 → "Robust to evaluated attacks (S1–S5)"로 변경, RoboGuard/Static Geofence 행 추가.

---

## E. 주장 수정 목록 (tex 라인 → 수정)

| # | 위치 | 현재 | 수정 |
|---|---|---|---|
| E-1 | L67 (기여1) | "the guarantee survives a fully compromised command pipeline" | "the guarantee is independent of the correctness of upstream command-generation and planning layers, provided the COE preconditions and the trusted stop path of Table~\ref{tab:tcb} hold" |
| E-2 | L47 (초록) | "without trusting upstream components" | "while treating all command-generation and planning components as untrusted" (TCB는 신뢰함을 암시적으로 유지) |
| E-3 | L47 (초록) | "confirming cross-platform transfer" | "consistent with the formula-predicted margin on a second platform" |
| E-4 | L116 (Table I) | "Attack Robust" ✓/✗ 열 | "Robust to evaluated attacks (S1–S5)"; 각주에 평가 범위 명시 |
| E-5 | L463 | "Its 40 false positives are safe goals that happen to fall within the margin... by design" | FP에서 제외하고 별도 카테고리로: "Goals outside the zone but inside the designed margin (within-margin probes) are rejected by construction; we report them separately from false positives, which are now zero. The margin staircase (Table X) shows each method rejecting exactly the probes inside its own margin." → `statistical_analysis.md`의 margin-probe 표 사용 |
| E-6 | L482 | "bypassing the control channel entirely" | "operating independently of the nominal controller output, while relying on the trusted safety-priority stop channel" |
| E-7 | L512, L536 | "PETSE's covariance-based margin absorbed the offset" | "the configured uncertainty envelope covered the evaluated bounded offset (0.5 m < M); arbitrary stealthy spoofing remains outside the guarantee unless independently detected" |
| E-8 | L504, L536 | "outcomes are deterministic across all 140 runs" / "deterministic 0/20 vs 20/20 ... clear evidence" | "all 20 trials per condition were mitigated; the 95% upper confidence bound on the per-condition failure probability is 13.9% (rule of three), and larger-scale hardware testing is needed to tighten it" — 시뮬레이션 CI(0/300, ≤1.0%)와 병기 |
| E-9 | L72 (기여3) | "ROS-compatible" / "A drop-in ROS node" | "ROS 2" / "a single ROS 2 lifecycle node" — L550, L559 포함 문서 전체에서 "ROS"→"ROS 2" 통일 (리뷰어 2) |
| E-10 | L544 | "does not yet exist" | D 항목 참조 문장으로 교체 |
| E-11 | L546 | S1 하드웨어 누락 사유 "LLM refused" | 프롬프트 시도 횟수·모델·수동 goal 사용 여부 보고 (리뷰어 3) + ④단계에서 S1 하드웨어 실험 후 문단 교체 |
| E-12 | L328 유지 | Remark 2 (sufficient vs necessary) | 유지 — 이미 좋은 방어. COE Definition과 상호 참조 추가 |

표기 체크리스트 (리뷰어 2): "ROS 2" 통일(E-9), "AMCL" 철자 — tex 본문엔 ACML 없음, **그림 파일(attack_scenarios_s1s4_v2.jpg, overview.jpg) 내 텍스트 확인 필요**. GitHub 재현 저장소 링크를 §V Setup에 추가.

---

## F. 시나리오 번호 매핑 (논문 ↔ 코드)

논문 S1–S4는 코드 S1–S5에서 salami(코드 S2)를 **의도적으로 제외**하고 앞으로 당긴 번호다. 제외 근거: S2 salami는 no_guard를 제외한 모든 방법이 통과(5/5 TP)해서 방법 간 변별력이 없음 (`statistical_analysis.md`의 S2 포함/제외 표 비교로 확인됨). 논문 번호는 유지한다.

| 논문 | 내용 | 코드 |
|---|---|---|
| S1 | Direct hazard goal | S1 |
| S2 | Path inducement | S3 |
| S3 | Velocity + delay | S4 (approved_then_deviate 계열) |
| S4 | Position spoofing | S5 (TOCTOU) |
| — | Salami (변별력 없어 논문 제외) | S2 |

표 재생성 시 **"S1+S3+S4+S5" (S2 제외) 테이블이 논문 기준**이다 — statistical_analysis.md의 첫 섹션. 통계 스크립트의 시나리오 라벨은 코드 기준이므로 논문 반영 시 번호 치환 필요.

salami의 활용처: 비교 실험이 아니라 (a) LLM 특화 위협의 존재 증명(Threat Model 서술 1–2문장 + 각주/supplementary), (b) "모든 gate 있는 방법이 다단계 우회를 막는다"는 부수 결과. 논문 본문 시나리오로 되살리지 않는다.

수치도 전면 교체 대상: 구표(1,920 trials, F1 0.917, CBF-Adaptive TP 98)는 roboguard 추가 + margin-probe 분리 + S4 잔재 제거 후 무효. 새 소스는 `experiment_results/gazebo_s1_s6/statistical_analysis.md` (PETSE Recall 100% [98.1,100], F1 1.000, VR 0/300 rule-of-3 ≤1.0%, McNemar 전부 p<0.001 Holm 보정 후).

---

## G. 남은 작업과의 연결

- ② S4/S5 스푸핑 재설계: C-3의 bounded-spoof 위협 모델을 뒷받침할 magnitude sweep (0.1–1.0 m, stealthy 포함)
- ④ S1 하드웨어: E-11 문단 교체
- 통계 표 반영: Table II–IV를 statistical_analysis.md 기반으로 재생성 + CI/유의성 각주

---

## H. ③ 좁은 복도 + 인접/중첩 구역 (R3 대응, 완료)

**실험**: `run_geometry_stress.py` — 실제 enforcement 코어(`geofence_core.GeofencePolicy`의 `evaluate_point`/`evaluate_segment`)를 직접 구동. 재구현이 아니라 Algorithm 1 그대로. 결과 `geometry_stress_results.json`, 그림 `figures/geometry_stress.png`.

### H-1. 좁은 복도 false rejection (R3 지적 3)
두 벽 사이 중앙선 직선 경로. gap 스윕 결과:
- PETSE는 **gap > 2M = 1.124m일 때만 통과** (중앙선 clearance > M). 전환점이 정확히 2M.
- **보수 구간(gap 0.6–1.124m)**: 로봇 footprint clearance는 양수(0.08–0.34m, 실제로 안 부딪힘)인데도 REJECT — R3가 말한 "false rejection". 단 이는 **불확실성 마진 예산** 안이므로 설계된 보수성이며 ε로 조절 가능.
- gap 0.4m만 진짜 위반(footprint clearance −0.02m, 로봇이 벽에 닿음).

**논문 서술**: "PETSE는 자유폭이 2M 미만인 복도를 거부한다. 이 중 로봇 몸체가 물리적으로 통과 가능한 구간(footprint clearance>0)의 거부는 불확실성 마진을 가용성과 맞바꾼 결과이며, 좁은 산업 통로에서는 ε를 낮춰(마진 축소) 통과율을 높일 수 있다." Limitations L546의 "narrow corridors ... remains future work"를 이 정량 결과로 교체.

### H-2. 인접/중첩 구역 — 합집합이지 합산 아님 (R3 지적 4)
R3의 핵심 오해("두 구역의 팽창 경계가 겹치면 마진을 합산해 로봇을 차단하지 않는가?")를 코드로 반박:
- 측정된 자유폭이 **union 예측 max(0, s−2M)과 정확히 일치**, sum 예측(s−4M)과 불일치.
- 중점은 **한 구역의 마진 안에 있을 때만(s ≤ 2M)** 거부. 마진은 더해지지 않음.
- 예: s=2.5m에서 union은 1.375m 자유폭을 남기지만, (틀린) sum 모델이면 0.251m. 실제 코드는 union을 따름.

**논문 반영**: Method 섹션에 union 수식 $F^+(t) = \bigcup_i (F_i \oplus B(M(t)))$ 명시(재포지셔닝 C의 §IV 목록에 이미 포함) + 이 실험을 근거로 "인접 구역이 마진을 합산해 과차단하지 않음"을 1문장+그림으로 답변. `evaluate_point`가 zone별 독립 검사 후 첫 위반에서 반환하는 구조가 union 의미론을 보장.

**주의**: `evaluate_segment`는 `get_safety_margin()`(velocity=None→v_max) 사용, `evaluate_point`는 `compute_margin(velocity=v_max)` 사용 — 둘 다 v_max 기준이라 일관됨. 단 실제 Nav2 실행 시 goal gate가 어느 경로를 쓰는지 코드 정합성 재확인 필요(E-11/Algorithm 1 정합성과 연동).

---

## I. Additive vs. RSS 마진 비교 (R1/AE "과보수적·중복계산" 대응, 완료)

**실험**: `run_margin_comparison.py` — 논문 `monte_carlo_formula_validation`의 `SimParams`/물리 재사용, 조건당 2M trials. 결과 `margin_comparison_results.json`, 그림 `figures/margin_comparison.png` (3-패널).

성분: est=0.412, track=lat=brake=0.050. **Additive=0.562, RSS=0.421, 비율=1.335** (논문 Remark 2의 1.33과 정확히 일치).

리뷰어의 "additive가 과보수·중복계산"에 대한 **2단 반박**:

**반박 1 — additive는 자의적 보수가 아니라 99.9 백분위와 일치한다.**
독립(nominal) 오차에서 실측 변위의 경험적 백분위 마진과 비교:
| 마진 | 값(m) | nominal 위반% | 적대적 위반% |
|---|---|---|---|
| static (zσ만) | 0.412 | 1.75% | 100% |
| RSS | 0.421 | 1.52% | 100% |
| empirical p99 | 0.446 | 1.00% | 100% |
| empirical p99.9 | 0.565 | 0.10% | 0% |
| **additive (PETSE)** | **0.562** | **0.11%** | **0%** |

→ **Additive(0.562) ≈ empirical p99.9(0.565)**. 즉 additive 마진은 임의의 과보수가 아니라 실제 변위 분포의 99.9 백분위를 겨냥하며, ε=0.003 설계 목표(0.1% 위반)와 정확히 부합. "중복계산"이 아니라 원하는 tail quantile로 보정된 것.

**반박 2 — RSS는 적대적 위협에서 완전히 실패한다 (33% 추가분은 낭비가 아니라 적대적 강건성의 값).**
모든 오차원이 같은 방향·최대치로 정렬되는 적대적 조건(논문 Def.1/Prop.1)에서:
- static/RSS/empirical-p99: **100% 위반** (완전 붕괴)
- empirical-p99.9/additive: **0% 위반**

→ RSS는 "덜 보수적"이 아니라 위협 모델 하에서 **안전하지 않음**. Additive의 +33%는 독립 가정을 공격자가 깰 수 있기 때문에 지불하는 정확한 minimax 비용. 이것이 Prop 1(tightness)의 실증.

**논문 반영**: Remark 2 뒤에 이 표+그림 추가. "additive는 (i) 독립 오차에서 p99.9와 일치하는 보정된 마진이고, (ii) 적대적 정렬에서 RSS/percentile이 100% 실패할 때 유일하게 0% 위반을 유지한다"로 서술. §IV의 alternative RSS formulation 언급과 연결. 이걸로 R1 지적 2 + AE "안전 마진 설계 정당화" 동시 해소.

---

## J. CBF 파라미터 민감도 (R1.6 + AE "CBF 튜닝 불공정" 대응, 완료)

**실험**: `run_cbf_sensitivity.py` — 프로젝트의 실제 CBF 구현(`safety_baselines.CBFSpatialSafety`, γ·δ 파라미터)을 직접 구동. (γ∈{0.1,0.5,1,2,5,10}) × (δ∈{0.1,0.3,0.45,0.562,0.75}) = 30 config 스윕. 결과 `cbf_sensitivity_results.json`, 그림 `figures/cbf_sensitivity.png`.

리뷰어의 "CBF 민감도 분석·최적 보정 없음 → 비교 불공정"에 대한 반박:

**발견 1 — 위반율은 δ에만 좌우되고 γ와 무관.**
히트맵의 각 행(δ 고정)이 γ 전체에서 동일. 이유는 코드에서 직접 확인됨: `CBFSpatialSafety.evaluate`는 `h = signed_dist(goal) − δ ≥ 0`으로 ALLOW 판정 → **goal-gate가 δ에만 의존, γ는 런타임 QP 보수성에만 영향**. 리뷰어의 "δ 민감" 지적은 맞지만, 우리는 이미 δ를 스윕했고 CBF-Adaptive에 PETSE 마진(0.562)을 부여함.

**발견 2 — 최적 튜닝 CBF도 33% 위반 (PETSE보다 큰 마진 줘도).**
30개 config 중 최선(γ=0.1, δ=0.75, **PETSE 0.562보다 큰 마진**)도 2/6=33% 위반. δ=0.562(=CBF-Adaptive)는 50% 위반.

**발견 3 — 잔여 실패는 구조적, 파라미터로 못 고침.**
| 시나리오 | best-tuned CBF | 원인 |
|---|---|---|
| S1 (goal) | 0/3 ✓ | δ로 해결 가능(파라미터) |
| S4 (spoof 0.5m) | 0/1 ✓ | δ=0.75>0.5라 흡수 |
| **S2 (path)** | **1/1 ✗ (전 config)** | goal-gate가 경로 미검사 — 구조적 |
| **S3 (channel)** | **1/1 ✗ (전 config)** | 제어입력을 loop 내부에서 필터 → 채널 침해 시 우회 — 구조적 |
- S3는 **깨끗한 채널이면 0 위반**(CBF 보장 성립) → 실패 원인이 γ가 아니라 채널 침해임을 확인. S2/S3(침해)는 30개 config **전부** 실패.

**논문 반영**: Setup의 CBF-Adaptive 서술 + 새 민감도 그림/문단. "우리는 30개 CBF (γ,δ) 설정을 스윕했고, PETSE보다 큰 마진을 준 최적 CBF도 33% 위반이 남는다. 이는 CBF가 (i) goal만 검사하고 (ii) 제어 loop 내부에서 동작하기 때문이며, 어떤 파라미터로도 제거되지 않는 구조적 한계다"로 서술. 이것이 독창성(R1.1) 논거(PETSE는 outside-loop + path-aware)도 강화. R1.6 + AE "CBF 튜닝" 해소. Table I의 CBF 각주에 γ,δ 스윕 범위 명시.

---

## K. ε 선택 Pareto + 선택 규칙 (R3 + AE "ε 선택 지침 부재" 대응, 완료)

**실험**: `run_epsilon_pareto.py` — ε가 유일한 위험 노브(M_est=z_{1-ε}·σ). ε→M(ε)→위반율은 논문 monte_carlo 물리(3M trials/ε)로, 가용성 손실은 ③ 좁은 복도 결과(free=W−2M)로 모델링. 결과 `epsilon_pareto_results.json`, 그림 `figures/epsilon_pareto.png`.

**핵심 수치** (3m 산업 통로 기준):
| ε | M(m) | 위반율 | 가용성 손실 |
|---|---|---|---|
| 0.0001 | 0.708 | 0.003% | 47.2% |
| 0.001 | 0.614 | 0.033% | 40.9% |
| **0.003 (기본)** | **0.564** | **0.10%** | **37.6%** |
| 0.01 | 0.499 | 0.39% | 33.3% |
| 0.05 | 0.398 | 2.19% | 26.5% |
| 0.1 | 0.342 | 4.91% | 22.8% |

**운영자 선택 규칙**: $\varepsilon^* = \arg\min_\varepsilon [C_{\mathrm{viol}} P_{\mathrm{viol}}(\varepsilon) + C_{\mathrm{block}} P_{\mathrm{block}}(\varepsilon)]$
| 비용비 $C_{viol}:C_{block}$ | ε* | M | 위반율 | 차단율 |
|---|---|---|---|---|
| 1000:1 (안전 최우선) | 0.0001 | 0.70 | 0.003% | 46.7% |
| 100:1 | 0.0008 | 0.62 | 0.026% | 41.4% |
| 10:1 (균형) | 0.008 | 0.51 | 0.31% | 33.9% |
| 1:1 (가용성 우선) | 0.1 | 0.34 | 4.91% | 22.8% |

**검증**: Gazebo epsilon_multi(281 trials)의 위반율이 ε에 따라 단조 증가(그림 (a) 보라 점) — probe battery라 이산적이지만 MC 곡선 방향 확인.

**논문 반영**: 새 "Operator guidance" 소절 + Pareto 그림. "기본값 ε=0.003은 0.1% 위반 대 통로폭 38% 소비의 knee에 위치하며, 안전-임계 배치는 ε를 낮추고(예 1000:1 → ε≈10⁻⁴), 처리량-우선 배치는 높인다(1:1 → ε=0.1). 비용비만 정하면 선택 규칙이 ε*를 준다"로 서술. R3 "ε 선택" + AE "ε 분석" 해소.

---

## L. Fail-stop 후 recovery (R3 "통로 막힘/복구불가" 대응, 완료)

**실험**: `run_recovery_policies.py` — 실제 `GeofencePolicy._project_to_safe`(safe holding point)를 재사용한 2D kinematic 시뮬. 로봇이 통로(y=1.7, 폭 1.4m)를 주행 중 존 옆에서 **일시적 COE 고장**(공분산 스파이크 / 속도·지연 주입, 8s)이 runtime monitor를 발동. 결과 `recovery_policies_results.json`, 그림 `figures/recovery_policies.png`.

**정책 비교 (모두 위반 0):**
| 고장 | 정책 | 완료 | 위반 | 통로차단(s) |
|---|---|---|---|---|
| covariance | P1 fail-stop | ✗ STUCK | 0 | 110 |
| covariance | P2 hold-resume | ✓ | 0 | 8.1 |
| covariance | P3 retreat-replan | ✓ | 0 | 0.1 |
| covariance | P4 reduced-speed | ✓ | 0 | 8.1 (hold 폴백) |
| velocity/latency | P1 fail-stop | ✗ STUCK | 0 | 110 |
| velocity/latency | P4 reduced-speed | ✓ | 0 | **0 (무정지 통과)** |

**핵심 메시지**:
1. **fail-stop 단독 = R3 우려 실현**: 통로를 무기한(110s=잔여 전 구간) 막고 임무 실패. 하지만 위반 0 — fail-stop은 degradation ladder의 안전한 최후단.
2. **recovery 추가 = 안전 유지하며 가용성 복원**: P2/P3/P4 모두 임무 완료, 위반 0. retreat(P3)가 통로를 가장 빨리 비움(0.1s).
3. **정직한 뉘앙스 — 고장 유형별 적절 recovery가 다름**: 감속(P4)은 M_est가 속도 무관이라 **공분산 고장은 못 고치고**(hold 폴백), M_lat/M_brake가 v에 비례하는 **속도·지연 고장만 무정지 통과** 가능. 이 구분이 recovery 설계의 원리.

**논문 반영**: Limitations의 "fail-stop aborts the mission entirely; a softer degradation path ... is not yet implemented"(L548)를 이 실험으로 교체. 재포지셔닝 C-2의 Level 3(Safe Degradation) ladder를 구체화: normal → reduced-speed → retreat/replan → controlled stop → e-stop. "fail-stop은 최후 fallback이며 상위 단계가 대부분 가용성을 복원함(위반 0 유지)"으로 서술. retreat 궤적 자체가 PETSE 제약을 만족(`_project_to_safe`)하므로 recovery 중에도 안전 불변.
```
