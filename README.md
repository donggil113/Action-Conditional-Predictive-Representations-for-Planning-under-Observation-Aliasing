# P5 — Action-Conditional Predictive Representations for Planning under Observation Aliasing (가제)

**현재 상태:** `TECHNICAL_TEST_PASS` + `SCIENCE_NOT_EVALUATED`. 자세한 내용은 [STATUS.md](STATUS.md)에 있다.
과학적 결론은 아직 없다. 파일럿은 사전등록만 했고 승인·실행되지 않았다.

- 연구 질문, 반증 조건, 비교군 정보 접근: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)
- 선행연구 대조 (claim table): [docs/PRIOR_ART.md](docs/PRIOR_ART.md)
- AI 사용 내역: [AI_USAGE.md](AI_USAGE.md)

## 구성 (Python 표준 라이브러리만 사용)
numpy, torch, pytest가 설치되어 있지 않다. 지시에 따라 설치하지 않았고, 순수 Python으로 구현했다 (blocker는 STATUS 참조).

| 파일 | 내용 |
|---|---|
| `acpr/env.py` | Aliased T-maze (privileged simulator, snapshot/restore). symmetric / asymmetric(대조) 변형 |
| `acpr/accounting.py` | `TransitionLedger` / `MeteredSimulator`: 모든 transition(main·branch·eval)을 실행 전에 과금하고, cap을 넘는 step은 실행하지 않는다. restore는 별도로 세고 평가에서는 금지한다 |
| `acpr/data.py` | actor가 볼 수 있는 `History`/`Episode`/`Branch`, evaluator 전용 `EvaluatorLabel` |
| `acpr/collect.py` | budget 안에서 branching 수집. RNG를 분리해 branching이 main trajectory를 바꾸지 않게 한다 |
| `acpr/splits.py` | train/dev/calibration/test의 서로 겹치지 않는 seed stream, 내용 중복률 |
| `acpr/autodiff.py` | vector 단위 reverse-mode autodiff, Adam, MAC 계수기 |
| `acpr/models.py` | Elman RNN encoder, latent rollout predictor (ACP: action one-hot / MSP: 0 벡터, 파라미터 수 동일) |
| `acpr/train.py` | k-step window (종료 뒤는 pad하고 supervise, 절단 뒤는 mask), 학습 루프 |
| `acpr/evaluate.py` | 동결 encoder + stage-2 reward head + 전수 MPC planner, forced-commit 정확도, clue probe |
| `acpr/run.py` | CPU runner (manifest, raw log, FAILED/NOT_RUN 보존, 파일럿 승인 게이트) |
| `acpr/analyze.py` | 사전등록한 paired bootstrap 분석과 중단 규칙 플래그 |

## 실행
```bash
# 전체 테스트 (stdlib unittest)
python3 -m unittest discover -s tests -t . -v

# smoke run (기술 점검 전용; 수치는 과학적 결과가 아님)
python3 -m acpr.run --config configs/smoke.json --out runs/<run_id>

# 사전등록 파일럿: pilot_approved=false 이므로 runner가 거부함 (exit 2)
python3 -m acpr.run --config configs/pilot_memory_prereg.json --out runs/<run_id>

# 사전등록 분석 (파일럿 결과가 생긴 뒤)
python3 -m acpr.analyze --prereg configs/pilot_memory_prereg.json --runs runs/<run_id> ...
```

각 run 디렉터리에는 `config.json`, `raw_log.jsonl`, `metrics.json`, `manifest.json`이 들어 있다. manifest에는 git commit/dirty, source/config/data sha256, transition ledger, 환경, wall/CPU 시간, peak RSS, 총 MAC가 기록된다. arm별 encoder/head 가중치 sha256, 파라미터 수, MAC, 실지출 transition, restore 횟수는 metrics.json에 있다.
