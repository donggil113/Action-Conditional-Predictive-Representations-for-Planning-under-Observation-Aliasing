# 선행연구 대조 (claim table)

확인일: 2026-09-26. 원자료(영문, 인용문과 fetch 실패 기록 포함)는 [prior_art_raw_en.md](prior_art_raw_en.md)에 있다.

**검증 표기**
- `FULL_TEXT_VERIFIED`: 원문 PDF나 HTML의 해당 절을 읽었다.
- `FULL_TEXT_VERIFIED(HTML)`: fetch 도구의 HTML 추출본으로만 읽었고, PDF와 대조하지 않았다.
- `FULL_TEXT_UNVERIFIED`: 초록이나 메타데이터만 확인했다.

아래 인용 중 TRPO, James & Singh, TD-MPC2, DBC, PBL, PSD, Littman et al.은 추출 텍스트에서 grep으로 다시 대조했다. Kwon et al.은 메인 세션에서 arXiv abs 페이지와 HTML을 직접 한 번 더 확인했다.

## 1. 가장 가까운 원논문 8편과 이번 계획과의 관계

| # | 논문 | 검증 | 이미 알려진 것 | 이번 계획과의 관계 |
|---|---|---|---|---|
| 1 | Littman, Sutton, Singh (2001), *Predictive Representations of State*, NeurIPS 14 (proceedings 페이지 저자 표기는 Littman, Sutton) | FULL_TEXT_VERIFIED | 상태 = action-conditional·multi-step test 예측의 벡터. 유한 POMDP는 상태 수 이하의 linear PSR을 가진다. aliasing 예시(float/reset)는 k-order Markov로 모델링할 수 없다 | **핵심 메커니즘 자체가 PSR이다.** "action-conditional 예측이 history를 요약한다"는 주장은 신규가 아니다 |
| 2 | Singh, James, Rudary (2004), *PSRs: A New Theory…*, UAI | FULL_TEXT_VERIFIED | system-dynamics matrix, history-conditional test 예측 p(t\|h), 선형 차원 | symmetric 변형에서 marginal 예측이 불충분하다는 것은 이 이론의 직접적 귀결이다 |
| 3 | Barreto et al. (2017), *Successor Features…*, NeurIPS (+ Dayan 1993: 메타데이터만) | FULL_TEXT_VERIFIED | MDP 가정. 예측은 action sequence가 아니라 **고정 정책**에 조건화된다 | 정책-marginal 예측 계열이다. 우리 MSP 기준선과 개념상 가깝다 |
| 4 | Zhang et al. (2021), *DBC*, ICLR | FULL_TEXT_VERIFIED | "partial-observability를 명시적으로 다루지 않고" stacked frame을 상태로 근사한다. Block MDP, 1-step latent dynamics | task-oriented 표현 자체는 기여로 주장하지 않는다 |
| 5 | Hansen, Su, Wang (2024), *TD-MPC2*, ICLR | FULL_TEXT_VERIFIED | "infinite-horizon MDPs". 단일 관측 encoder, action-conditional latent dynamics, H-step consistency + reward + value 손실 | **multi-step action-conditional latent 예측은 이미 표준이다.** TD-MPC2는 history encoder가 없고 POMDP를 다루지 않으므로, 그대로 official baseline으로 쓸 수 없다 |
| 6 | Hafner et al. (2019) PlaNet, ICML / DreamerV3 (arXiv 2023; Nature 2025 제목 *Mastering diverse control tasks through world models*) | FULL_TEXT_VERIFIED (DreamerV3는 arXiv v2만) | POMDP 틀의 recurrent world model(RSSM), latent overshooting(multi-step), 재구성 손실 | recurrent world model 자체는 기존 기술이다. 우리 GRU/Elman 구현은 이들의 재현이 아니다 |
| 7 | Subramanian, Sinha, Seraj, Mahajan (2022), *AIS*, JMLR | FULL_TEXT_VERIFIED | history 압축 Z가 (Z, A)로 보상과 다음 AIS/관측을 예측하면 근사 DP 가치손실 한계가 성립한다. LSTM 실험 | action-conditional·history 기반 예측 목적 + planning 보장의 이론이 이미 있다 (1-step) |
| 8 | Ni et al. (2024), *Bridging State and History Representations*, ICLR | FULL_TEXT_VERIFIED | POMDP history encoder의 self-predictive 목적(ZP/OP/RP)을 AIS와 연결한다. 1-step이며 planning은 없다 | history 기반 self-predictive 목적의 분석이 이미 있다 |

## 2. 근접 후속·관련 연구 (요약)

| 논문 | 검증 | 핵심 |
|---|---|---|
| **Kwon, Yang, Nowak, Hanna (2024, v2 2025)**, *An Empirical Study on the Power of Future Prediction in Partially Observable Environments*, arXiv 2402.07102 | FULL_TEXT_VERIFIED(HTML) | **가장 가까운 선행연구.** history encoder φ(h)와 action-conditional k-step 관측 예측 ψ(o₁:ₖ do a₁:ₖ \| φ(h))를 쓰고 PSR test로 설명한다. POPGym 기억 과제를 쓴다. test action은 "에피소드당 무작위 시점 하나"에 실제로 실행되며 simulator branching은 없다. action-free ablation은 HTML에서 찾지 못했다 |
| Gregor et al. (2019), SimCore, NeurIPS | FULL_TEXT_VERIFIED | belief RNN에서 "agent의 action만 보며" 앞으로 시뮬레이션하는 action-conditional multi-step 예측이다. 자기 궤적으로만 학습한다 |
| Guo et al. (2018), *Neural Predictive Belief Representations*, arXiv 1811.06407 | FULL_TEXT_VERIFIED (초록 원문; 구조는 HTML) | CPC\|Action. "multi-step predictions and action-conditioning are critical"이라는 결과로, **action-conditional과 action-free의 경험적 비교가 이미 있다** (DMLab, contrastive) |
| Guo et al. (2020), PBL, ICML | FULL_TEXT_VERIFIED | 압축 history에서 미래 latent로의 forward action-conditional 예측 (DMLab-30) |
| Khetarpal et al. (2025), *A Unifying Framework for Action-Conditional Self-Predictive RL*, AISTATS | FULL_TEXT_VERIFIED(HTML) | BYOL-AC(action별)와 BYOL-Π(정책-marginal)의 이론 비교. MDP, 1-step |
| Venkatraman et al. (2017), PSD, NeurIPS / Downey et al. (2017), PSRNN, NeurIPS | FULL_TEXT_VERIFIED | RNN 상태에서 미래 k개 관측을 예측하며 future action 조건이 없다. **우리 MSP 기준선과 구조가 같은 선행 형태**이다 (추출한 수식 기준; PSD의 controlled 변형은 미확인) |
| Sutton & Tanner (2005), TD networks; Zheng et al. (2021) | FULL_TEXT_VERIFIED (Zheng은 HTML) | action-conditional 예측 질문을 **실행된 action**으로만 학습한다 |
| Oh et al. (2015); Schwarzer et al. (2021) SPR; Schrittwieser et al. (2020) MuZero; Grimm et al. (2020) | FULL_TEXT_VERIFIED | action-conditional multi-step 예측·unroll. SPR은 frame stack을 쓰고, MuZero·value equivalence는 MDP 또는 상태를 가정한다 |
| Schulman et al. (2015), TRPO | FULL_TEXT_VERIFIED | vine: "restored to particular states, … only possible in simulation". **simulator restore branching의 선례**이다 (정책경사 추정용) |
| James & Singh (2004), ICML | FULL_TEXT_VERIFIED | null history로 reset한 뒤 h를 재생하고 test를 실행해 p(t\|h)를 추정한다. **같은 history에서 여러 test 결과를 수집하는 선례**이다. 결정적 T-maze에서는 우리 save/restore와 사실상 동등하다 |
| Ni et al. (2023), *When Do Transformers Shine in RL?*, NeurIPS; Ni et al. (2022), ICML; Samsami et al. (2024), R2I, ICLR | FULL_TEXT_VERIFIED (Ni 2022는 HTML) | Passive/Active T-Maze 정의, recurrent model-free 기준선, 기억 과제용 world model |
| McCallum (1996); Whitehead & Ballard (1991); Lambrechts et al. (2022); Hefny et al. (2018); Yin et al. (2023); Mhammedi et al. (2024); Lambrechts et al. (2024); Teoh et al. (2025) | FULL_TEXT_UNVERIFIED | 메타데이터나 초록만 확인했다 (perceptual aliasing, RPSP, reset 기반 탐색·이론, informed POMDP, next-latent 예측) |

## 3. 주장 구분표

| 이번 프로젝트의 요소 | 판정 | 근거 |
|---|---|---|
| future prediction으로 history 표현 학습 | **기존** | PSR, PSD, PSRNN, SimCore, PBL, Kwon et al. |
| action-conditional multi-step 예측 | **기존** | PSR, TD nets, SimCore, CPC\|Action, PBL, SPR, TD-MPC2, MuZero, Kwon et al. |
| action-conditional이 action-free보다 낫다는 경험적 결과 | **기존 (다른 설정)** | Guo et al. 2018 (DMLab); Khetarpal et al. 2025 (이론, MDP) |
| symmetric aliasing에서 marginal 예측이 불충분하다는 점 | **기존 이론의 특수 사례** | PSR (Littman 2001; Singh 2004). 우리 테스트는 구성을 확인할 뿐이다 |
| task-oriented / value-equivalent 표현 | **기존** | DBC, value equivalence, MuZero, AIS |
| recurrent world model | **기존** | PlaNet/Dreamer, R2I |
| simulator restore / reset으로 같은 history에서 여러 행동 결과 수집 | **기존 기법** | TRPO vine, James & Singh 2004, Mhammedi et al. 2024 |
| 신경망 history encoder의 action-conditional 예측 target을 **branching으로 수집하고 branch 비용을 예산에 포함**하는 조합 | 확인한 논문에서 **찾지 못함** | 신규성 인증이 아니다. 검색은 망라적이지 않고 기존 기법의 조합이다 |
| 같은 encoder·용량·데이터에서 ACP vs MSP를 aliased T-maze의 **planning 결정 정확도**로 비교 | 확인한 논문에서 **찾지 못함** | 신규성 인증이 아니다. Kwon et al. HTML에서 action-free ablation은 보지 못했으나 PDF와 부록은 대조하지 않았다 |

**결론:** 기여로 주장할 수 있는 여지는 좁다. 기껏해야 "branching 특권과 그 비용을 명시적으로 회계한 통제 비교"라는 **실험 설계상의 기여**이며, 그 가치도 파일럿의 S1/S2 결과(`docs/RESEARCH_PLAN.md`)에 달려 있다. 파일럿 전에는 어떤 신규성도 주장하지 않는다.
