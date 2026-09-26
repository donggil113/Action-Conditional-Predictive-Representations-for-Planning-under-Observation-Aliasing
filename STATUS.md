# STATUS (2026-09-26)

| 구분 | 상태 |
|---|---|
| 소프트웨어 | **TECHNICAL_TEST_PASS**: unittest 63/63 통과, skip 0. smoke run 2회 모두 `TECHNICAL_RUN_COMPLETE` |
| 과학 | **SCIENCE_NOT_EVALUATED**: 파일럿 미실행. smoke 수치는 과학적 결과가 아니다 |
| 파일럿 | **PREREGISTERED_NOT_APPROVED**: `configs/pilot_memory_prereg.json`, `pilot_approved=false`이며 runner가 거부한다 (exit 2) |
| 신규성 | 인증하지 않음. [docs/PRIOR_ART.md](docs/PRIOR_ART.md) 기준으로 핵심 구성요소는 모두 기존 연구에 있다 |
| 평가 판별력 진단 (2026-09-26 2단계) | **HEADROOM_PRESENT** (기술 진단, 합성 환경). 9/9 cell OK. 요약은 §7, 상세는 [docs/DIAG_EVAL_DISCRIMINABILITY.md](docs/DIAG_EVAL_DISCRIMINABILITY.md) |

소프트웨어 측면에서 파일럿을 실행할 수는 있다. 그러나 이는 신규성 인증이나 채택 가능성 확인이 아니며, 아래 blocker와 위험을 먼저 검토해야 한다.

## 1. 이번 단계 요구사항 대비
| 요구 | 상태 | 위치 |
|---|---|---|
| 선행연구 5–8편 + 후속 대조 | 완료. 핵심 8편은 FULL_TEXT_VERIFIED, 일부 후속은 UNVERIFIED로 표시 | `docs/PRIOR_ART.md` |
| 연구 질문, 반증 조건, 정보 접근, 목적함수 가정 고정 | 완료 | `docs/RESEARCH_PLAN.md` |
| 과거 clue가 다르고 현재 관측이 같으며 미래 action 결과가 다른 환경 | 완료. symmetric과 asymmetric(대조) 변형 | `acpr/env.py`, `tests/test_env.py` |
| ground truth는 collector/evaluator 전용, actor에는 history만 | 완료. 시그니처, 필드 whitelist, 직렬화, spy 테스트 | `acpr/data.py`, `tests/test_information_access.py` |
| simulator branching, 모든 branch transition을 예산에 포함 | 완료. hard cap, 실지출 = cap, restore 별도 계수 | `acpr/accounting.py`, `acpr/collect.py`, `tests/test_accounting.py` |
| 작은 recurrent encoder + action-conditional predictor | 완료. Elman RNN + latent rollout, 유한차분 gradient 검사 | `acpr/models.py`, `acpr/autodiff.py`, `tests/test_autodiff.py` |
| 같은 history·capacity·branch experience의 multi-step predictive 기준선 | 완료. MSP는 같은 데이터·초기화·파라미터 수를 쓰고 action 입력만 0이다. (2단계 명확화: MSP는 **action-free ablation**이다. 명목상 파라미터 수가 같다고 해서 강한 action-conditional baseline과 표현력이나 유효 학습 capacity가 같은 것은 아니다. action 열 hidden×\|A\|개는 gradient를 받지 않는다) | `tests/test_information_access.py::TestArmParity` |
| 정보 접근, episode split, transition 회계, horizon 테스트 | 완료 | `tests/test_{information_access,splits,accounting,horizon}.py` |
| 최소 CPU runner | 완료 | `acpr/run.py`, `configs/smoke.json` |

## 2. 실행 기록 (실제 명령과 결과)
환경: Python 3.11.15 (CPython), Linux x86_64, CPU 4개, GPU 미사용, 서드파티 패키지 없음.

| 커밋 | 명령 | 결과 | wall | peak RSS |
|---|---|---|---|---|
| 8cf9bd1 | `python3 -m unittest discover -s tests -t . -v` | 50/50 OK | <1s | – |
| 8cf9bd1 | `/usr/bin/time -v python3 -m acpr.run …` | **실행 안 됨**: `/usr/bin/time` 없음 (exit 127). run은 시작되지 않았고 빈 로그 파일은 삭제함 | – | – |
| 8cf9bd1 | `python3 -m acpr.run --config configs/smoke.json --out runs/smoke_20260926T151203Z_8cf9bd1` | 8/8 OK. **SUPERSEDED**: probe 붕괴, 층화 전. 보존함 | 4.63s | 20.2MB |
| d72e769 | 같은 테스트 명령 | 61/61 OK | ~1s | – |
| d72e769 | `python3 -m acpr.run --config configs/smoke.json --out runs/smoke_20260926T151604Z_d72e769` | 10/10 OK (5 arm × 2 변형) | 9.51s | 20.3MB |
| d72e769 | `python3 -m acpr.run --config configs/pilot_memory_prereg.json --out …` | REFUSED (exit 2), 디렉터리 생성 안 됨 | – | – |
| e5afde0 | 같은 테스트 명령 | 63/63 OK | 5.06s | – |
| 5c08935 | 같은 테스트 명령 (문서만 변경) | 63/63 OK | 5.22s | – |

smoke run 2의 source sha256은 현재 `acpr/`와 같다 (`461cedc9…`). 테스트 로그는 `runs/test_logs/`에 있다.

### smoke run 2의 기술적 관찰 (과학적 해석 금지)
- 실지출 transition은 모든 arm에서 800 = cap이다. branch arm의 restore는 312회, nobranch arm은 0회다.
- 같은 transition cap에서도 계산비용은 다르다. stage-1+2 MAC는 branch arm 6.1M, nobranch arm 9.1M이다 (window 수 차이).
- forced-commit 정확도는 모든 arm에서 정확히 0.500이다. smoke 설정(3 epoch)에서는 stage-2 head가 학습되지 않는다. positive control(`tests/test_evaluator.py`)에서 정보가 있는 표현은 30 epoch에 1.0에 도달하므로, 평가 경로 자체는 작동한다.
- **random encoder의 clue probe가 1.00**이다 (L≤3). 짧은 corridor에서는 학습 없이도 clue가 선형으로 남는다. → 위험 R-1.
- asymmetric `*_nobranch`는 full-action planner timeout 1.00이다. 덜 학습된 head에서 "FORWARD 후 arm"을 과대평가하는 미루기 현상이다 (positive control 디버깅으로 재현).
- test의 junction 도달 prefix 관측열 중 63.3%가 train에 존재한다. 작은 환경이라 seed가 분리되어도 내용이 겹친다.

## 3. 결정과 변경 로그 (모두 파일럿 데이터 없이 이루어짐)
1. 저장소가 비어 있었다: 커밋, 파일, 원격 브랜치가 없었다. 보존할 기존 산출물, ARCHIVE_METHOD, 보류 결정은 없었다.
2. numpy, torch, pytest가 없어 stdlib만으로 구현하고 `unittest`를 썼다. 설치하지 않았다.
3. smoke 1 이후: probe feature를 표준화했다 (다수 클래스로 붕괴). 평가 clue를 evaluator가 층화하도록 했다 (base rate 잡음). `msp_nobranch` arm(2×2 완성)과 lr grid를 추가했다.
4. smoke 2와 positive control 이후: primary metric을 full-action success에서 forced-commit accuracy로 바꿨다. 사전등록 stage-2 epoch을 10에서 20으로 올렸다. 근거는 §2와 `docs/RESEARCH_PLAN.md` §4다.

## 4. Blocker
| 항목 | 내용 | 영향 |
|---|---|---|
| numpy/torch/pytest 미설치 | 지시대로 설치하지 않음. `uv`는 존재 (`/root/.local/bin/uv`) | 순수 Python은 느리다. 작은 기억 과제 파일럿은 가능하지만(추정 ~4.2 CPU-h, 미검증) **continuous control 확장은 torch 설치 승인 없이는 불가능하다** |
| `/usr/bin/time` 없음 | runner 내부의 `resource.getrusage`와 `perf_counter`로 대체 | 없음 |
| 선행연구 fetch 실패 | psr.pdf 503, TD-net PDF 503, ACM DL/OpenReview 403 등 (`docs/prior_art_raw_en.md`) | 대체 출처로 확인. Kwon et al.은 HTML만 확인했고 PDF·부록은 미대조 |
| 라이선스 | 저장소에 LICENSE 없음. 외부 코드, 데이터, 가중치, pretrained encoder는 **사용하지 않음** | 공개·배포 전에 소유자가 라이선스를 결정해야 한다 |

## 5. 미검증 주장과 위험
- **R-1:** random RNN 표현이 이미 clue를 보존할 수 있다. 그러면 H1이 지지되더라도 "학습이 기억을 만든다"는 뜻이 아니다. 파일럿의 R1 비교로 점검하며, 실패하면 환경을 재설계하기 전에는 주장하지 않는다.
- **R-2:** symmetric 변형에서 H1은 PSR 이론으로 이미 예측된다 (구성상 MSP 목적은 clue와 독립). 지지되어도 새 발견이 아니다.
- **R-3:** 파일럿 자원 추정(~4.2 CPU-h)은 1 epoch micro-benchmark에서 외삽한 값이다.
- **R-4:** 수치 fixture 테스트(유한차분, positive control)는 구현 검증이지 증명이 아니다.
- **R-5:** "branching으로 수집하고 비용을 회계한 조합"은 확인한 논문에서 찾지 못했을 뿐이다. 검색은 망라적이지 않으며, James & Singh (2004)의 reset 기반 수집과 매우 가깝다.
- 낮은 예측 손실이 임의 정책의 장기 planning을 보장한다고 주장하지 않는다.

## 6. 다음 단계 (각각 별도 승인 필요)
1. 파일럿 승인: `pilot_approved=true`로 새 커밋을 만든 뒤 seed별 병렬 실행 (`--seeds`), 이어서 `python3 -m acpr.analyze …`.
2. positive pilot이고 **STOP_EXPANSION 플래그가 없을 때만** partial-observation continuous control로 확장을 검토한다. 이때 torch 설치 승인이 필요하다.

## 7. 2단계: random encoder와 downstream 평가의 판별력 진단 (인수 기준 `6fbfe7f`)
**승인 범위:** 제한된 CPU 실행으로 이 단계 전체 1,800 CPU-s 이하. 기존 full pilot(`pilot_approved=false`)과 continuous-control 확장은 승인되지 않았고 실행하지 않았다. 설치와 다운로드는 없다.

### 변경 (코드는 영어, 기존 테스트는 수정하지 않음)
- `acpr/controls.py`: history-only informative control, memoryless control, train-split 전용 standardizer (다른 split을 넣으면 거부).
- `acpr/train.py`: `on_epoch_end` hook. epoch 후보를 한 run의 checkpoint에서 얻는다. `acpr/models.py`: `copy_predictor`.
- `acpr/evaluate.py`: forced-commit 동률 수와 기댓값 정확도 진단을 추가했다. 기존 `forced_commit_accuracy`와 RNG 소비는 바뀌지 않는다.
- `acpr/diagnose.py`: `RLIMIT_CPU`/`RLIMIT_AS`로 cap을 강제하고, cell 전에 비용을 검사하며, CAP_EXCEEDED·NOT_RUN·OOM·FAILED를 보존한다. 사전 고정 판정 규칙을 적용한다.
- `tests/test_diagnose.py`: 회귀 테스트 12개. checkpoint = 독립 짧은 run, 순서·인덱스 독립성, train 전용 표준화, memoryless는 어떤 head로도 정확히 0.5, 동률 처리, 판정 규칙, subprocess에서의 cap 강제를 검사한다.
- smoke 결과(random probe 1.0, corridor 2–3)는 이전 단계에서 이미 보았고, 이번 진단 설계의 동기가 되었다.

### 실행 기록
| 커밋 | 명령 | 결과 | CPU (user+sys) |
|---|---|---|---|
| 6fbfe7f | timing-only benchmark (scratchpad, 지표 미출력) | 첫 시도는 `ModuleNotFoundError` (PYTHONPATH 누락). 재시도에서 stage-2 epoch 1.58s/10k를 측정 | 0.02s + 3.51s |
| (커밋 전) | `python3 -m unittest discover -s tests -t .` | OK. **시간 미측정**, 약 5s로 추정 | ~5s |
| (커밋 전) | `python3 -m unittest tests.test_diagnose -v` | 12/12 OK | 2.03s |
| (커밋 전) | 전체 테스트 | 75/75 OK | 7.19s |
| 281020c | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m acpr.diagnose --config configs/diag_eval_discriminability.json --out runs/diag_20260926T160006Z_281020c` | **HEADROOM_PRESENT**. OK 9, FAILED/OOM/CAP_EXCEEDED/NOT_RUN 0. wall 1,199.5s, peak RSS 26.9MB | 1,188.6s |
| – | 결과 요약·교차확인 스크립트 (읽기 전용) | – | ~0.1s |
| 5b8e2ac | `python3 -m unittest discover -s tests -t . -v` (최종) | 75/75 OK, skip 0 (`runs/test_logs/unittest_5b8e2ac.log`) | 6.99s |

- 이 단계의 누적 CPU는 최종 테스트를 포함해 **약 1,215s**로, 상한 1,800s 이하다. 미측정 1회(~5s)는 추정치다.
- run이 진행되는 동안 stop-hook 요구에 따라 **미완성 산출물 스냅샷을 3번 WIP 커밋**했다 (06d22fa, 95908bc, 6aa899b). 최종 파일은 run 종료 뒤 5b8e2ac에 커밋했으며, 원로그를 덮어쓰지 않고 이어 쓴 것이다.

### 결과 요약 (forced-commit 정확도, seed 1000/1001/1002)
- informative control: 1.000/1.000/1.000 (full-action 성공 1.0, timeout 0)
- random recurrent (학습하지 않음): 0.500/0.500/0.500. clue probe test는 0.47/0.53/0.57
- memoryless: 0.500/0.500/0.500. full-action에서는 seed 1001/1002가 timeout 1.0 (stall)
- 판정 입력: informative − random = 0.50 (모든 seed), random − memoryless = 0.00

### 해석 범위
- **평가 경로는 작동한다.** 위험 R-1(random이 이미 천장)은 pilot 설정과 표준화 경로에서는 관찰되지 않았다.
- 그러나 학습된 encoder가 이 여지를 채운다는 증거는 없다. ACP/MSP checkpoint가 없어 NOT_AVAILABLE이고 재학습하지 않았다.
- 현재 환경은 주 benchmark 후보로 **유지할 수 있다** (`CURRENT_BENCHMARK_NONDISCRIMINATIVE` 아님). 다만 informative control이 1.0이어서 두 학습 encoder를 서로 비교할 때는 천장 효과가 생길 수 있다.
- full-action 성공률은 clue를 모르는 두 encoder를 0.5와 0.0으로 갈랐다. 표현이 아니라 commit/stall 행동에 좌우되므로 표현 비교 지표로 쓰면 안 된다.
- 실행하지 않은 것: asymmetric 변형, 표준화하지 않은 head 입력(pilot 사전등록 경로 그대로), ACP/MSP 참고 평가. 모두 NOT_RUN으로 기록했다.
- seed 1000–1002의 test split은 이제 개발 자료다.

### 남은 blocker와 결정 사항 (승인 필요)
1. 후속 비교(작은 규모)를 할지. 질문은 action conditioning 자체가 아니라 같은 경험·비용에서 남는 효과(S1, S2)여야 한다.
2. pilot stage-2에 train 표준화를 넣을지. 넣지 않으면 이번 판별력 결과가 그 경로에 그대로 적용되지 않는다.
3. 천장 효과 대책.
4. 순수 Python 성능 한계: cell당 약 130 CPU-s. 규모를 키우려면 torch 설치 승인이 필요하다.

