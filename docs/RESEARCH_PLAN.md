# 연구 계획 (고정본 v1, 2026-09-26)

> 상태: **SCIENCE_NOT_EVALUATED**. 이 문서는 파일럿 실행 **전에** 고정한 질문·반증 조건·비교 규칙이다.
> 파일럿은 승인되지 않았고 실행되지 않았다 (`configs/pilot_memory_prereg.json`, `pilot_approved=false`).

## 1. 연구 질문 (하나)

**RQ.** 관측 aliasing이 있는 작은 기억 과제(Aliased T-maze)에서 다음을 모두 동일하게 둔다.
- history 입력 (o, a, r)
- encoder·predictor 용량과 초기 가중치
- 학습 데이터: branch를 포함해 같은 transition 집합이며, 총 환경 transition 수도 같다
- 튜닝 범위와 평가 절차

이 조건에서, **action-conditional multi-step 예측 목적(ACP)** 으로 학습한 recurrent history 표현이 **action-marginal multi-step 예측 목적(MSP)** 으로 학습한 표현보다 aliased junction에서의 결정 정확도를 최소관심효과(MIE) 이상 높이는가? 결정 정확도는 두 표현을 동결한 뒤 같은 방식으로 학습한 action-conditional reward head와 같은 planner로 잰다.

- **H1 (primary):** symmetric 변형에서 `forced_commit_accuracy(acp_branch) − forced_commit_accuracy(msp_branch)`의 seed 평균이 **≥ 0.10**이고, 95% paired bootstrap CI 하한이 **> 0**이다 (10 seeds).
- **반증 조건:** 95% CI 상한이 0.10 미만이면 `NOT_SUPPORTED`이다 (MIE 크기의 효과가 배제됨). 그 외는 `INCONCLUSIVE`이다. FAILED·NOT_RUN seed가 있어 10쌍이 안 되면 `INCOMPLETE`이며, 좋은 seed만 골라 채우지 않는다.

### 이 질문이 새롭지 **않은** 부분 (사전 인정)
- symmetric 변형에서는 action-marginal 미래 분포가 clue와 **구성상 독립**이다 (`tests/test_env.py::test_symmetric_action_marginal_future_is_clue_independent`가 정확 열거로 확인). 따라서 MSP 목적은 clue를 보존할 gradient 유인을 갖지 않는다. 이는 PSR의 "test는 action-conditional이다"라는 기존 이론에서 바로 예측되는 결과다. H1이 지지되더라도 **새 이론적 발견이 아니다.**
- future prediction, multi-step latent prediction, task-oriented / self-predictive representation, recurrent world model, simulator state-restore branching은 모두 선행연구에 있다 (`docs/PRIOR_ART.md`). 이 프로젝트는 이들 중 어느 것도 신규 기여로 주장하지 않는다.
- 파일럿이 실제로 답하는 경험적 질문은 좁다.
  1. 유한 학습에서 MSP가 random 초기화 RNN이 이미 가진 clue 정보를 **실제로 지우는가**. smoke run에서 random encoder의 clue probe는 1.00이었다 (`R1`로 점검).
  2. ACP의 이점이 simulator restore 특권(branching) 없이도 남는가 (`S2`).
  3. 같은 실지출 transition에서 branching이 비용만큼의 값을 하는가 (`S1`).

## 2. 비교군과 정보 접근 (모두 이 저장소에서 직접 구현했으며 official baseline이 아니다)

| arm | encoder 입력 | predictor의 action 입력 | 학습 데이터 | simulator restore | 학습 transition (실지출) | stage-1 target | stage-2 head | evaluator label 접근 |
|---|---|---|---|---|---|---|---|---|
| `acp_branch` | o₀..oₜ, a₀..aₜ₋₁, r₁..rₜ | onehot(aₜ₊ⱼ) | branch 포함 | 사용 (횟수 기록) | B (정확히 소진) | o, r (k-step) | 동일 | 없음 |
| `msp_branch` | 동일 | **0 벡터** | `acp_branch`와 **동일 데이터** | 동일 | B | 동일 | 동일 | 없음 |
| `acp_nobranch` | 동일 | onehot | branch 없음 | 0 | B | 동일 | 동일 | 없음 |
| `msp_nobranch` | 동일 | 0 벡터 | `acp_nobranch`와 동일 | 0 | B | 동일 | 동일 | 없음 |
| `random_frozen` (참조) | 동일 | — (stage-1 없음) | stage-2는 branch 데이터 | 사용 | B | — | 동일 | 없음 |

- **용량:** encoder 파라미터 수는 모두 같다. predictor 파라미터 수는 ACP와 MSP가 같지만, MSP는 `Wd`의 action 열(hidden×|A|)이 gradient를 받지 못하므로 `effective_param_count`로 따로 보고한다.
- **초기화:** 같은 seed 안에서는 encoder, predictor, stage-2 head 초기 가중치가 arm 간에 같다.
- **계산비용:** 같은 transition cap이라도 실제 비용은 다르다. nobranch arm은 window 수가 많아 MAC가 약 1.5배다 (smoke 실측 9.1M vs 6.1M). MAC와 wall-clock을 arm별로 기록하며, "동일 cap"을 "동일 실지출"로 부르지 않는다.
- **추가 supervision:** 없음. ACP와 MSP는 같은 window와 같은 target(o, r)을 쓰고, 다른 점은 predictor에 action이 들어가는지뿐이다.
- **Ground truth:** clue, 위치, RNG 상태는 `AliasedTMaze`/`EvaluatorLabel`에만 있다. actor·학습·planner 함수 시그니처에는 들어가지 않으며, 테스트로 검사한다 (`tests/test_information_access.py`).

## 3. 목적함수의 가정
- **ACP:** window (h_t, a_{t:t+k−1}) → Σ_j [CE(o_{t+j+1}) + λ·(r̂−r)²]. 학습되는 것은 조건부 분포 p(o | h_t, a_{t:t+j})이다. **가정:** window 안의 action은 hidden state에 의존하지 않는다. 이를 위해 behavior는 History만 받고, continuation은 uniform이다. hidden state에 의존하는 action(예: oracle expert)이 섞이면 action 조건화가 confounding된다.
- **MSP:** 같은 target을 action 없이 예측하므로, 데이터의 action 혼합(behavior + branch uniform)에 대한 marginal p(o | h_t)를 학습한다. 따라서 **behavior policy에 의존한다.**
- **종료 처리:** 종료 이후는 흡수 기호 `OBS_TERMINAL_PAD`와 보상 0으로 채우고 supervise한다 (종료는 actor가 볼 수 있다). timeout이나 budget으로 잘린 부분은 mask한다.
- **보상:** 제곱오차(λ=1). stage-2는 reward 항만 쓴다.
- **주장하지 않는 것:** 낮은 prediction loss/MSE가 임의의 정책이나 장기 planning을 보장한다고 주장하지 않는다. 평가 대상은 이 과제, 이 planner, 이 horizon뿐이다.

## 4. 평가 절차 (모든 arm 동일)
1. Stage 1 (arm별 목적)을 학습하고, lr은 고정 grid {0.003, 0.01}에서 **각 arm 자신의 dev 목적함수**로 고른다.
2. encoder를 동결하고, 새 action-conditional reward head를 같은 train window, 같은 하이퍼파라미터로 학습한다.
3. Test 에피소드는 evaluator가 clue를 교대로 배정해 균형을 맞춘다 (상수 정책이면 정확히 0.5). scripted FORWARD로 junction까지 간다.
4. **Primary:** `forced_commit_accuracy`. 첫 junction 도달 시 planner가 {LEFT, RIGHT} 중 최적 arm을 고르는 비율이며, 환경 step을 추가로 쓰지 않는다.
5. **Secondary:** full-action MPC `success_rate`, `timeout_rate`, clue probe(calibration split에서 적합, 표준화한 logistic, evaluator 전용), stage-2 dev reward MSE.
6. 평가 ledger는 snapshot/restore를 **금지**한다. 숨긴 rollout이 불가능하다.

> **사후 변경 기록 (파일럿 데이터 없음):** smoke run 1 뒤에 test clue를 층화했고 probe feature를 표준화했다. smoke run 2와 positive-control 점검 뒤에 primary를 full-action success에서 forced-commit으로 바꿨다. 근거는 head가 덜 학습되면 planner가 "FORWARD 후 arm"을 과대평가해 junction에서 무한히 미루는 현상이다. 같은 근거로 stage-2 epoch을 10에서 20으로 늘렸다. 모든 변경은 파일럿 실행 전에 했다.

## 5. 사전 고정값 (`configs/pilot_memory_prereg.json`)
- seeds 0–9, 변형 {symmetric, asymmetric}, corridor 3–6, distractor 4, horizon k=3, plan horizon 2, γ=0.9
- train budget 10,000 transitions (branch 포함), dev 2,000, calibration 200 에피소드, test 200 에피소드
- hidden 16, stage-1 20 epochs, stage-2 20 epochs, batch 16 에피소드, Adam, grad clip 5
- MIE 0.10 (forced-commit accuracy, 절대값), α=0.05, bootstrap 10,000 (seed 12345)
- dev split은 모든 arm이 **같은** branch 포함 데이터(2,000 transitions)를 쓴다. nobranch arm에게는 학습 분포와 약간 다르다는 점을 알고 둔 선택이다.
- 자원 상한: 총 8 CPU-hours, 프로세스당 wall 3h, peak RSS 2GB. runner가 자동으로 집행하는 것은 프로세스당 wall과 RSS뿐이다. 총 CPU-hours는 여러 프로세스의 manifest `process_cpu_s` 합으로 운영자가 확인한다. 추정 소요 ~4.2 CPU-h는 micro-benchmark에서 외삽한 값이며 **미검증**이다. 상한을 넘은 arm은 NOT_RUN으로 기록되고, 분석에서는 INCOMPLETE가 된다.

## 6. 비교와 중단 규칙
| 이름 | 비교 | 역할 |
|---|---|---|
| H1 | symmetric: acp_branch − msp_branch | primary |
| S1 | symmetric: acp_branch − acp_nobranch | 같은 실지출에서 branching의 값 |
| S2 | symmetric: acp_nobranch − msp_nobranch | restore 특권 없이도 이점이 남는가 |
| C1 | asymmetric: acp_branch − msp_branch | 메커니즘 대조 (marginal도 clue를 담는 조건) |
| R1 | symmetric: acp_branch − random_frozen | 학습이 random 표현보다 나은가 |

- H1은 SUPPORTED인데 S2가 SUPPORTED가 아니면 **STOP_EXPANSION**이다. 이점이 simulator restore(숨긴 rollout 비용)에 의존하기 때문이다.
- R1이 SUPPORTED가 아니면 환경이 학습된 기억을 요구하지 않는 것이므로, 환경을 재설계하기 전에는 어떤 주장도 하지 않는다.
- C1도 비슷한 크기로 SUPPORTED이면 효과가 action 조건화에 특이적이지 않다. 메커니즘 주장을 하지 않는다.
- 추가 history, oracle, 숨긴 rollout 비용으로만 이기면 자동 확장을 중단한다.
- partial-observation continuous control로의 확장은 positive pilot **및** 별도 승인 뒤에만 한다.
