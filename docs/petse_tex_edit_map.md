# PETSE 논문 tex 편집 지도 (재투고본)

대상 파일: `~/Downloads/PETSE__..._8_ (2)/conference_101719_8p.tex` (566줄, TII 투고본).
근거: `docs/petse_repositioning_plan.md`(§A–L) + 실험 산출물. 줄 번호는 원본 기준.

**전역 원칙 3가지**
1. LLM 로봇 보안 → **runtime assurance layer**로 재포지셔닝 (계획 §A).
2. 구식 수치 전면 교체: 1,920 trials / F1 0.917 / CBF-Adaptive TP 98 → 새 통계(`statistical_analysis.md`). **논문 S1–S4 = 코드 S1/S3/S4/S5** (salami=코드S2 제외 유지). 논문표는 `statistical_analysis.md`의 "S1+S3+S4+S5 (S2 제외)" 섹션 사용.
3. margin-probe 재프레이밍: near/mid_boundary 거절은 FP가 아니라 별도 카테고리 → PETSE **FP=0, F1=1.000**.

범례: 🔴교체필수 🟡수정 🟢추가 ⚪표기

---

## 1. 제목 · 초록 · 키워드

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🔴 L34–35, L39 | "PETSE: Probabilistic Execution-Time Safety Envelope for LLM-enabled Mobile Robots" | 제목 교체(택1): *An Uncertainty-Aware Runtime Safety Envelope for Mobile Robots under Compromised Navigation Commands* (LLM 유지 시 §A 3안) | §A |
| 🔴 L46–48 | 초록: "any prompt...", "1,920 Gazebo trials...F1=0.917", "confirming cross-platform transfer" | ① 첫 문장 LLM→"untrusted upstream command sources (an LLM interface being one)"로 완화. ② 수치 새 통계로: N trials, PETSE VR 0/… (rule-of-3 ≤1.0%), recall 100% [98.1,100]. ③ "confirming"→"consistent with the formula-predicted margin on a second platform" (E-3). ④ additive의 ε-보정·적대적 강건성 1구 추가 | §A, `statistical_analysis.md`, §E-3/I |
| ⚪ L50–52 | keywords: "LLM-enabled robotics" 포함 | "runtime assurance", "runtime safety monitor" 추가; LLM 키워드 유지 가능 | §A |

---

## 2. Introduction (L55–79)

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🟡 L58–62 | LLM 공격면 중심 도입 | 유지하되 프레이밍: "LLM은 upstream 명령을 손쉽게 오염시키는 대표 인터페이스이며, PETSE는 명령 소스 종류와 무관하게 실행계층에서 방어"로 1–2문장 조정 | §A |
| 🔴 L64 | "the guarantee survives a fully compromised command pipeline" 취지 | 삭제/완화 예고 (실제 완화는 L67) | §E-1 |
| 🔴 L65–77 | Contributions 4개 (기여1 overclaim, 기여3 "ROS-compatible") | **계획 §B의 4개 기여로 통째 교체**: ①risk-parameterized margin(ε), ②scoped COE guarantee, ③three-gate arch(ROS 2 lifecycle), ④systematic adversarial eval(통계+HW). "four attack scenarios" (salami 제외) | §B |
| 🟢 L64 근처 | (없음) | 독창성 방어 3줄(§A: ε-유도 마진 / 3-gate 커버리지 / cbf_inflated 격리) 삽입 | §A |
| ⚪ L79 | 섹션 안내 "Section III... S1–S4" | 시나리오 수·번호 확인 후 유지 | — |

---

## 3. Related Work (L82–130)

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🟢 L102–107 근처 | 통제이론·공간안전 소절 | **신규 소절 "Runtime assurance & execution-layer monitors"**: Simplex/RTA(Sha 2001), Nav2 Collision Monitor, RoboGuard, reachability 모니터 인용+한계 서술 | §D (문헌조사 필요) |
| 🟡 L109–128 Table I | 6행, "Attack Robust" 열, Static Geofence만 execution | ① 열명 "Attack Robust"→"Robust to evaluated attacks (S1–S4)" + 각주. ② RoboGuard 행 추가. ③ Static Geofence 유지 | §D, §E-4 |
| 🔴 L130 | "No prior method that we are aware of derives a dynamic execution-layer margin..." | 대상 좁힘: "...derives an execution-layer margin **from an operator-specified violation tolerance ε jointly with** covariance, tracking, latency, braking" | §D |

---

## 4. Threat Model (L133–151)

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🔴 L136–139 | 시스템/위협 서술, "not assumed attacker can actuate motors or rewrite firmware" | **§C-1 TCB 신뢰표 신규 삽입** + "PETSE does not guarantee safety if both the navigation stack and the independent stop path are compromised" 명시 | §C-1, R1-5/AE-3 |
| 🟢 L139 근처 | (없음) | bounded-spoof 명시: ‖δ_spoof‖ ≤ Δ_spoof = M_est+M_track (=0.462m). stealthy spoof는 보장 밖 | §C-3, ②실험 |
| 🟡 L150 (S4) | "S4 Position Spoofing" | 지속-스푸핑 정의 명확화 + Δ_spoof 참조. covariance 흡수 주장 제거(→결과 L512서 수정) | §C-3 |

---

## 5. Method (L153–371)

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| ⚪ L172–179 | "ROS 2 lifecycle node" (이미 ROS 2) | 본문 전역 "ROS"→"ROS 2" 통일(L72 이미 수정, L550 "standard ROS node", L559 "ROS navigation stack") | §E-9/R2-2 |
| 🟢 L253 근처 | Geometric Interpretation | union 수식 F⁺(t)=⋃ᵢ(Fᵢ⊕B(M(t))) 추가 (인접 구역 과차단 방지 명시) | §H-2, R3-5 |
| 🟢 L262 앞 | Safety Guarantee 시작 | **§C-2 Certified Operating Envelope(COE) 정의 삽입** + 안전주장 3단계(certified/detection/fail-stop) 문단 | §C-2, R1-3/R1-4 |
| 🟢 L323–325 (Remark 2 뒤) | 보수성 비율 1.33 서술 | **additive vs RSS 표+그림 참조 추가**: additive≈p99.9, 적대적서 RSS 100% 실패 | §I, `margin_comparison.png` |
| 🟡 L327–329 (Remark 3) | sufficient vs necessary | 유지 + COE 정의 상호참조 | §C-2 |
| 🟡 L335–371 Alg.1 | 경로 교차 시 REJECT | 유지하되 각주: "footprint 통과 가능한 마진접촉 경로도 거부(보수성, ε 조절)" + HW 정합성 주석 | §H-1, §19(정합성) |

---

## 6. Experiments (L374–536) — 수치 전면 교체

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🟡 L378–388 Setup | 5 baselines, "35.6/143.6 µs" | ① RoboGuard 추가(6 baselines). ② **CBF 서술에 (γ,δ) 30-config 스윕 명시**: 최적도 33% 위반, 구조적. ③ GitHub 링크 추가. ④ 오버헤드 수치 재확인 | §J, R1-6; R2-3 |
| 🔴 L400–420 Table confusion | N=320, PETSE TP220/FP40, CBF-Adaptive TP98 | **`statistical_analysis.md` 값으로 교체**. margin-probe 분리 → PETSE FP=0. RoboGuard 행 추가. seeds=20 | `statistical_analysis.md` |
| 🔴 L422–440 Table detection | F1: PETSE 0.917, CBF-Adaptive 0.548 | **새 F1로 교체**: PETSE 1.000[1.000,1.000], recall 100%[98.1,100]. **CI 열 추가**. RoboGuard 행 | `statistical_analysis.md` |
| 🟢 L440 근처 | (없음) | **신규: 유의성 표** (McNemar exact + Holm, PETSE vs 각 baseline p<0.001; discordant 0) + 시드분산(PETSE 1.000±0.000) | `statistical_analysis.md` |
| 🔴 L442–461 Table VR | VR 값들 (구 데이터) | 새 VR로 교체 + **rule-of-3**: "0 violations in N trials; 95% upper bound ≤1.0%" (E-8) | `statistical_analysis.md`, §E-8 |
| 🟡 L463 | "Its 40 false positives are safe goals...by design" | **§E-5로 교체**: FP=0, near/mid_boundary는 margin-probe 별도 보고 + margin staircase 표 참조 | §E-5, `statistical_analysis.md` staircase |
| 🟢 L484 근처 | CBF-Adaptive vs PETSE 소절 끝 | **CBF (γ,δ) 민감도 그림 추가** + "PETSE보다 큰 마진 줘도 33% 위반=구조적" | §J, `cbf_sensitivity.png` |
| 🟢 L486–501 Ablation 뒤 | stress test만 | **신규 소절 3개**: (a) ε Pareto+선택규칙, (b) 좁은복도/인접구역 기하, (c) 지속-스푸핑 Δ_spoof | §K/H/②, 그림 3장 |
| 🟡 L503–536 Real-Robot | 140 trials, S1 없음, "deterministic 0/20...clear evidence", "covariance absorbed offset" | ① **E-8**: "20/20 mitigated; 95% upper bound 13.9% (rule-of-3), larger HW test needed". ② **E-7 (L512)**: covariance 흡수 주장→"configured envelope covered the evaluated bounded offset; stealthy spoof outside guarantee". ③ **S1 하드웨어 결과 추가**(④ 실기체 후) + 프롬프트 수/모델/수동goal 보고(R3-3) | §E-7/E-8, R1-8/R3-3 |

---

## 7. Limitations (L538–550)

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🔴 L544 | "does not pit PETSE against a peer execution-layer defense—one that...does not yet exist" | **삭제/교체**(자기모순): "RoboGuard·static geofence·CBF-Adaptive와 비교; 완전 독립 adaptive geometric monitor는 없어 CBF-Adaptive가 same-margin 대조" | §D, §E-10 |
| 🟡 L546 | narrow corridor/S1 HW "future work" | 좁은복도는 §H 결과로 대체 서술; S1 HW는 ④ 후 결과로 | §H, R1-8 |
| 🔴 L548 | "fail-stop aborts mission entirely; softer degradation not yet implemented" | **§L recovery 실험으로 교체**: degradation ladder + "recovery가 위반0 유지하며 가용성 복원" | §L, R3-7 |
| 🟡 L550 | "trusted enforcement layer...ROS node" | TCB 표(§C-1) 참조로 강화 + "ROS 2 node". stealthy spoof/펌웨어는 명시적 out-of-scope | §C-1 |

---

## 8. Conclusion (L552–561) · 기타

| 위치 | 현재 | 교체 액션 | 근거 |
|---|---|---|---|
| 🟡 L555–559 | "operates independently...LLM", "1,920 trials" | 수치 갱신 + "runtime assurance layer" 표현. "bypassing control channel"류 없으면 유지 | §A, 통계 |
| ⚪ L482 | "bypassing the control channel entirely" | "operating independently of the nominal controller output while relying on the trusted stop channel" | §E-6 |
| 🟢 전역 | 그림 참조 | 신규 그림 5장 삽입: `margin_comparison`, `cbf_sensitivity`, `epsilon_pareto`, `geometry_stress`, `spoof_budget_sweep`, `recovery_policies` | 산출물 |
| ⚪ 그림파일 | attack_scenarios/overview.jpg | 그림 내 텍스트 "ACML/ROS" 오타 확인·수정 (R2-2) | R2-2 |

---

## 9. 실행 순서 (권장)

1. **수치 인프라 먼저**: Table confusion/detection/VR를 `statistical_analysis.md`로 교체 + 유의성/CI 표 신규 → 나머지 본문이 이 수치를 참조.
2. **Method 골격**: COE 정의(§C-2) + TCB 표(§C-1) + union 수식 삽입.
3. **신규 결과 소절**: additive/RSS(§I), CBF 민감도(§J), ε Pareto(§K), 기하(§H), 스푸핑(②), recovery(§L) — 그림 5장.
4. **포지셔닝**: 제목/초록/Intro 기여/Related Work(§A/B/D).
5. **주장 완화**: §E-1~E-10 일괄 적용.
6. **Limitations/Conclusion** 갱신.
7. **미착수 대기**: S1 하드웨어(R1-8)·envelope DWB/RPP(R3-6)는 실험 후 해당 칸 채움. GitHub(R2-3) 저장소 정리.

## 10. 이 편집으로도 못 채우는 칸 (실험/조사 선행 필요)

- **S1 하드웨어**(L503–536, Limitations) — 실기체 실험 후에만 작성 가능.
- **envelope DWB/RPP 교차검증**(L232–237 식3 정당화) — Gazebo 실험 필요. 현재 tex는 "fitted from Gazebo"만 있음(R3-6 미해결).
- **Related Work 문헌 인용**(§D) — 실제 논문 조사·bib 추가 필요.
- **GitHub**(Setup) — 저장소 공개·README.
