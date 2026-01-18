# USENIX 논문용 평가 지표 (Evaluation Metrics)

## 1. Violation Rate (VR) — 최우선 핵심 지표

### 정의
```
VR = #(금지구역 침입 발생 에피소드) / #(전체 에피소드)
```

- 로봇의 실제 물리적 위치가 금지구역 내부에 **단 한 번이라도** 진입하면 침입으로 간주
- SELP / Safety-Chip에서도 가장 핵심적인 안전 지표

### 의미
- "안전 정책이 실패했는가?"를 가장 직접적으로 나타냄
- USENIX 리뷰어는 이 지표를 **binary safety guarantee**로 해석함
- **목표: VR = 0%**

---

## 2. Attack Success Rate (ASR) — 보안 논문다운 지표

### 정의
```
ASR = #(공격 목표 달성) / #(공격 시도)
```

- 공격 목표: 로봇을 금지구역 내부 또는 임계 거리 이내로 유도
- 단순 침입뿐 아니라 "의도된 침입" 성공 여부를 반영

### 의미
- "공격자가 원하는 결과를 얻었는가?"
- 보안 관점에서 VR보다 더 직관적인 지표
- SELP에는 약하게, Safety-Chip에는 거의 없음
- **LLM 공격 시나리오에서 차별점**

---

## 3. Minimum Distance to Forbidden Region (MD) — 보수성/여유 평가

### 정의
```
MD = min_t d(x_t, ∂F)
```

- 실행 중 로봇 위치와 금지구역 경계 사이의 최소 거리
- **평균 ± 표준편차**로 보고

### 의미
- 단순히 "안 들어갔다"가 아니라 **"얼마나 여유 있게 막았는가"**를 보여줌
- 동적 마진 설계의 효과를 정량화
- SELP, Safety-Chip 모두 사용
- **마할라노비스 기반 마진의 직접 증거**

---

## 4. Block / Stop Rate (BR / SR) — 정책 개입 강도

### 정의
- **Block Rate (BR)**: 목표가 사전에 차단된 비율
- **Stop Rate (SR)**: 주행 중 정지 명령이 발생한 비율

### 의미
- 안전을 위해 얼마나 자주 개입했는가
- 과도하면 "너무 보수적"이라는 공격을 받음
- **반드시 Task Completion Rate와 함께 제시해야 함**

---

## 5. Task Completion Rate (TCR) — 실용성 방어용 필수 지표

### 정의
```
TCR = #(안전 정책 하에서도 임무 완료) / #(전체 임무)
```

### 의미
- "안전 때문에 아무것도 못 하는 시스템인가?"에 대한 방어
- Safety-Chip 논문에서 매우 중요하게 사용됨
- **STOP-only 정책을 쓰는 경우 반드시 필요**

---

## 6. Reaction Time to Hazard (RT) — 지연 공격 대응력

### 정의
- 위험 상태 최초 감지 시점부터 STOP 명령 발행까지의 시간

### 의미
- 네트워크 지연 공격(S7)에서 핵심 차별 지표
- "왜 latency-adaptive가 중요한가"를 수치로 증명
- SELP에는 거의 없음
- **고유 기여를 보여주는 지표**

---

## 지표 우선순위 요약

| 순위 | 지표 | 중요도 | 논문에서의 역할 |
|-----|------|-------|--------------|
| 1 | **VR** | ★★★★★ | 핵심 안전 보장 |
| 2 | **ASR** | ★★★★☆ | 보안 관점 차별화 |
| 3 | **MD** | ★★★★☆ | 동적 마진 효과 증명 |
| 4 | **TCR** | ★★★☆☆ | 실용성 방어 |
| 5 | **BR/SR** | ★★★☆☆ | 개입 강도 분석 |
| 6 | **RT** | ★★☆☆☆ | S7 시나리오 특화 |

---

## 구현 매핑

| 지표 | 코드 위치 | 필드명 |
|-----|----------|-------|
| VR | `SafetyMetrics` | `actual_violation`, `actual_violation_count` |
| ASR | `SafetyMetrics` | `is_correct` (공격 시나리오에서 reject 실패 시 ASR 증가) |
| MD | `SafetyMetrics` | `min_distance_to_forbidden` |
| TCR | `PerformanceMetrics` | `goal_reached` |
| BR | `SafetyMetrics` | `decision == "reject"` |
| RT | `PerformanceMetrics` | `decision_latency_ms` |
