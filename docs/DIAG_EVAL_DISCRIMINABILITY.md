# 진단: random encoder와 downstream 평가의 판별력 (2026-09-26)

- **판정:** `HEADROOM_PRESENT` (사전 고정 규칙: 모든 seed에서 informative − random ≥ 0.10)
- **분류:** 합성 환경에서의 **기술 진단**이다. 실제 데이터 근거가 아니며, 표현학습 방법의 효과나 신규성에 대한 증거도 아니다.
- **Run:** `runs/diag_20260926T160006Z_281020c/` (`cells.json`, `verdict.json`, `manifest.json`, `raw_log.jsonl`, `config.json`)
- **Config:** `configs/diag_eval_discriminability.json`. 실행 전 커밋 `281020c`에 고정했다. config sha256은 `4cdba804…`, source sha256은 `19472a22…`다.

## 질문
현재 환경(pilot 사전등록 설정: corridor 3–6, distractor 4, k=3, symmetric)과 downstream 경로(동결 encoder → 새 action-conditional reward head → 전수 MPC planner)가 **표현학습의 추가 효용을 판별할 여지**를 갖는가?

## 설계 (세 encoder에 동일 적용)
| 항목 | 값 |
|---|---|
| encoder | A. `informative_control`: o₀의 clue 식별 + 현재 관측 one-hot. 허용된 history만 쓰고 ground truth는 없다<br>B. `random_recurrent`: 학습하지 않은 Elman RNN, pilot `random_frozen`과 같은 초기화<br>C. `memoryless_control`: 현재 관측 one-hot |
| head 입력 | 차원별 z-score. 통계는 train split의 history prefix에서만 구한다 |
| head | hidden 16, 초기화, 데이터, lr 0.01, batch 16을 A/B/C 모두 같게 둔다 |
| epoch 후보 | 20/40/80. 80 epoch 한 번의 checkpoint에서 **dev reward MSE**가 가장 낮은 것을 고르고, 동률이면 적은 쪽. test는 쓰지 않는다 |
| 데이터 (seed당, 세 encoder 공유) | train 10,000 transitions (branch 포함), dev 2,000, calibration 200 에피소드, test 200 에피소드 (clue 균형배치) |
| seed | 1000, 1001, 1002 (진단 전용, pilot seed 0–9와 분리) |
| 자원 | 1 프로세스, thread 변수 1, `RLIMIT_CPU` 1650/1670s, `RLIMIT_AS` 3GiB. cell별로 사전 비용 검사 |

## 결과 (forced-commit이 판정 지표이며 junction 결정 성능일 뿐 planning 성공률이 아니다)
| seed | A forced / full 성공 / timeout | B forced / full 성공 / timeout | C forced / full 성공 / timeout | B probe test | 선택 epoch A/B/C |
|---|---|---|---|---|---|
| 1000 | 1.000 / 1.000 / 0.00 | 0.500 / 0.500 / 0.00 | 0.500 / 0.500 / 0.00 | 0.47 | 80/20/80 |
| 1001 | 1.000 / 1.000 / 0.00 | 0.500 / 0.500 / 0.00 | 0.500 / 0.000 / 1.00 | 0.53 | 80/20/80 |
| 1002 | 1.000 / 1.000 / 0.00 | 0.500 / 0.500 / 0.00 | 0.500 / 0.000 / 1.00 | 0.57 | 40/80/80 |

- informative − random = **0.500**, random − memoryless = **0.000** (3 seed 모두)
- 평균 return: A 1.000, B 0.000, C 0.000
- forced-commit 동률: 0건. 동률을 기댓값 0.5로 처리한 정확도도 위 값과 같다.
- B와 C는 모든 test 에피소드에서 **같은 arm을 고정으로** 골랐다 (LEFT 200 또는 RIGHT 200). 그래서 균형배치에서 정확히 0.5가 나왔다. 에피소드 순번, seed, 균형배치 패턴이 새어 들어갔다면 clue를 모르는 encoder가 0.5를 넘었을 텐데, 그런 징후는 없다.
- dev reward MSE: A는 0.022–0.027. B와 C는 0.102–0.108로 서로 거의 같고, 20/40/80 사이 변화는 0.0015 이하다.
- probe (calibration에 적합한 logistic): A는 1.0/1.0, B는 calibration 0.55–0.60 / test 0.47–0.57, C는 0.5/0.5.

## 비용 (환경 transition과 계산을 따로 보고한다)
| 항목 | 값 |
|---|---|
| 환경 transition 합계 | 53,503. train main 10,103 + branch 19,897, dev main 2,044 + branch 3,956, calibration 3,286, test 14,217 |
| restore | train 9,888, dev 1,980, test 0 (평가 중 금지) |
| test transition | encoder마다 다르다. C는 seed 1001/1002에서 stall 때문에 2,501–2,505이고, 나머지 cell은 1,301–1,333이다. **같은 cap이 같은 지출을 뜻하지 않는다** |
| 계산 | cell당 CPU 130–135s, 1.45–1.48 G MAC. 총 13.17 G MAC |
| diagnostic 프로세스 | CPU 1,188.6s (user+sys), wall 1,199.5s, peak RSS 26.9MB |
| 설치·다운로드 | 없음 (0) |

## 이 결과로 말할 수 있는 것
1. **평가 경로는 작동한다.** 같은 head/planner가 clue 정보가 있는 표현으로는 1.0, 없는 표현으로는 정확히 0.5를 낸다. `EVALUATOR_UNRESOLVED`는 아니다.
2. **pilot 설정에서 random recurrent encoder는 memoryless 수준이다.** smoke(corridor 2–3, hidden 8)에서 본 "random probe = 1.0"은 이 설정에서 재현되지 않았다 (probe test 0.47–0.57). 따라서 STATUS의 위험 R-1(random이 이미 천장)은 **이 설정·이 경로에 한해** 관찰되지 않았다.
3. forced-commit 기준으로 R1의 MIE(0.10)를 판별할 여지(0.5)가 있다.
4. full-action 성공률은 clue를 모르는 두 encoder(B, C)를 0.5와 0.0으로 가른다. 즉 **표현의 clue 정보가 아니라 commit/stall 행동에 좌우된다.** full-action 지표를 표현 비교에 쓰면 안 된다는 근거다.

## 말할 수 없는 것
- 학습된 ACP/MSP encoder가 이 여지를 채운다는 증거는 없다. 기존 checkpoint가 없어 평가하지 않았고(NOT_AVAILABLE), 재학습도 하지 않았다.
- A는 손으로 설계한 참조이며, 학습 가능한 상한을 뜻하지 않는다. 두 학습 encoder가 모두 1.0에 가까우면 천장 효과로 서로를 구분할 수 없다.
- random의 실패가 "clue 정보가 전혀 없음"을 뜻하지는 않는다. 선형 probe(200 샘플)와 이 head로 **꺼내지 못했다**는 뜻이다. 비선형 정보나 더 많은 데이터·epoch에서의 결과는 미확인이다. B의 선택 epoch 중 하나는 최대 후보(80)였다. dev MSE는 평탄했지만, 더 긴 학습에서의 결과를 배제하지 못한다.
- 결과는 **train 표준화를 추가한 경로**에 대한 것이다. pilot 사전등록의 stage-2(표준화 없음) 경로는 실행하지 않았다 (NOT_RUN).
- asymmetric 변형은 실행하지 않았다 (NOT_RUN).
- 3 seed는 한 변형·한 설정에서의 반복일 뿐 독립 재현이 아니다. seed 1000–1002의 test는 이제 개발 자료다.
- 0.5/1.0이라는 극단값은 작은 결정론적 합성 환경의 성질이다. 다른 환경으로 일반화할 수 없다.

## 선행연구 경계
- 가장 가까운 비교 대상은 여전히 Kwon et al. (arXiv:2402.07102)이다. action-conditioned future prediction과, 표현학습을 RL과 분리하는 decoupled history learning을 다룬다.
- 이 저장소의 ACP는 그 논문의 공식 재현이 아니다.
- branching arm은 **resettable simulator**(임의 상태 save/restore)가 있어야 쓸 수 있다.

## 후속 후보 (실행하지 않음, 승인 필요)
- 판정상 random과 A 사이에 여지가 있으므로, 작은 후속 비교의 **후보**가 된다. 전체 pilot은 자동 실행하지 않는다.
- 후속 질문은 action conditioning 자체가 아니어야 한다. 동일 경험·비용에서 추가로 남는 효과를 물어야 하며, 예를 들면 기존 2×2의 S1(같은 실지출에서 branching의 값)과 S2(restore 없이 남는 ACP−MSP 차이)다.
- 그 전에 결정할 것:
  - (i) pilot stage-2에 train 표준화를 넣을지 (프로토콜 변경)
  - (ii) full-action 지표를 판정에서 뺄지
  - (iii) A가 1.0이라 생기는 천장 문제를 어떻게 다룰지
