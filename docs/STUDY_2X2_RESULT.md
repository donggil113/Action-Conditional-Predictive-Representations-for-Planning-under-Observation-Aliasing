# 2×2 제한 비교 결과 (2026-09-27)

- **판정:** `LEARNING_BENEFIT_NOT_SUPPORTED`. 사전 고정 규칙에 따르면 R1 = ACP_b − untrained이 모든 seed에서 MIE 0.10 이상이어야 하는데, 이를 충족하지 못했다.
- **분류:** 합성 환경의 관찰 결과다. 실제 데이터 근거도, 신규성 근거도 아니다.
- **run:** `runs/study2x2_20260926T225310Z_fd781be/` (`cells.json`, `verdict.json`, `manifest.json`, `raw_log.jsonl`, `weights/`)
- **config:** `configs/study_2x2_symmetric.json`. 실행 전 커밋 `fd781be`에 고정했다. config sha256 `a3c901c9…`, source sha256 `eb99ff66…`.

## 설계 요약
- symmetric maze. seed는 1, 2, 3으로, 사전등록 목록 중 평가 결과를 아직 보지 않은 첫 3개다. 독립 재현이 아니다.
- arm은 6개: A ACP+branch, B ACP+no-branch, C MSP+branch, D MSP+no-branch, E untrained, F informative control.
- pretraining은 수집 방식마다 10,000 transitions다. A와 C가 branch dataset 하나를, B와 D가 no-branch dataset 하나를 그대로 공유한다.
- readout pool은 restore-free이고 모든 arm이 공유한다: head_train 10,000, head_dev 2,000.
- pretraining 설정은 튜닝하지 않았다: 20 epoch, lr 0.01. head는 20/40/80 epoch checkpoint 중 dev reward MSE로 골랐다.

## 결과 (forced-commit; junction 결정일 뿐 전체 planning 성공률이 아님)
| arm | seed 1 / 2 / 3 |
|---|---|
| A ACP+branch | 0.500 / 0.500 / 0.475 |
| B ACP+no-branch | 0.500 / 0.500 / 0.500 |
| C MSP+branch | 0.500 / 0.500 / 0.500 |
| D MSP+no-branch | 0.500 / 0.500 / 0.500 |
| E untrained | 0.500 / 0.500 / 0.500 |
| F informative | 1.000 / 1.000 / 1.000 |

- 모든 paired contrast가 −0.025에서 0.000 사이다. 학습 arm이 모두 chance 바닥에 있어서 **branching 효과와 action conditioning 효과를 분리할 수 없다**. 효과가 0이라거나 둘이 동등하다는 증거가 아니다.
- 선형 probe(test)는 학습 arm에서 0.51–0.75, untrained에서 0.52–0.56이었다. 일부 학습 feature에 clue가 부분적으로 decode될 가능성이 있지만, 표본이 200이라 관찰일 뿐이다.
- full-action 지표에서는 stall에 의한 timeout이 최대 1.000까지 나왔다. 판정에는 쓰지 않았다.

## 비용
| 항목 | 값 |
|---|---|
| 환경 transition | 134,083. pretraining 60,000(restore 10,048), readout pool 36,000(restore 0), calibration 3,317, test 31,470(restore 0), split 검사용 test prefix 재생 3,296 |
| 계산 | pretraining은 branch가 0.62 GMAC / 56–60 CPU-s, no-branch가 1.19 GMAC / 104–108 CPU-s다. **같은 transition cap이 같은 compute를 뜻하지 않는다** |
| run | CPU 5,202 s, wall 5,242 s, 56.15 GMAC, peak RSS 42.0 MB. RLIMIT_CPU 7,000 s, RLIMIT_AS 3 GiB, thread 변수 1 |

## 설명 가능한 범위 / 불가능한 범위
- **가능:** 이 예산, 이 설정, 이 readout에서는 네 학습 조합 모두 untrained encoder보다 나은 junction 결정을 만들지 못했다.
- **불가능:**
  - branching이나 action conditioning의 효과 유무. 바닥 효과 때문이다.
  - 더 긴 학습이나 다른 readout에서의 결과.
  - Kwon et al.과의 비교.
  - 원인. 세 가지 가설을 원고에 CONJECTURE로 적었다.
