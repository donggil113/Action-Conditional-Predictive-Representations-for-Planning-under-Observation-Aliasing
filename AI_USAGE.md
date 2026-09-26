# AI 사용 내역

이 저장소의 현재 내용은 전부 Claude Code(Anthropic의 AI 코딩 에이전트)가 2026-09-26 한 세션에서 작성했다. 사람의 검토는 아직 없다.

| 항목 | AI 수행 내용 | 사람 검토 필요 사항 |
|---|---|---|
| 코드 (`acpr/`) | 환경, 회계, 수집기, autodiff, 모델, 학습, 평가, runner, 분석 전부 | 특히 `autodiff.py`의 gradient 정확성(유한차분 테스트로만 확인)과 `collect.py`의 회계 로직 |
| 테스트 (`tests/`) | 63개 unittest 작성·실행 | 테스트가 주장하는 성질이 연구 질문에 충분한지 |
| 설정 (`configs/`) | smoke 설정, 파일럿 사전등록 초안 | MIE 0.10, seed 수, 자원 상한, primary metric 선택의 타당성 |
| 문서 | README, STATUS, RESEARCH_PLAN, PRIOR_ART | 한국어 서술이 과장 없이 정확한지 |
| 선행연구 대조 | 하위 에이전트가 WebSearch/WebFetch로 서지와 원문 일부를 확인 | `FULL_TEXT_UNVERIFIED` 항목의 원문 확인, 누락된 근접 연구 |

## 원칙과 한계
- 실행하지 않은 결과를 서술하지 않았다. 수치는 모두 `runs/` 아래 실제 로그에서 나왔다.
- smoke run의 수치는 구현 점검용이며 과학적 결과가 아니다.
- 수치 fixture 테스트는 구현을 검증할 뿐 수학적 증명이 아니다.
- 선행연구 표는 AI가 웹에서 확인한 범위에 한정된다. 저자·연도·주장은 사람이 원문으로 다시 확인해야 한다.
- 패키지 설치, 데이터·가중치 다운로드, 유료 API, GPU는 사용하지 않았다.

## 2단계 (2026-09-26, 평가 판별력 진단)
- Claude Code가 담당한 것: `acpr/controls.py`, `acpr/diagnose.py`, train/models/evaluate의 작은 hook, `tests/test_diagnose.py`, 진단 config와 보고서.
- 사람 확인이 필요한 것: 판정 임계값(positive 0.90, negative ±0.10, MIE 0.10)과 설계 선택(train 표준화를 모든 encoder에 적용)이 타당한지.
- 하위 에이전트는 사용하지 않았다. 설치, 다운로드, 유료 API, GPU도 사용하지 않았다.

