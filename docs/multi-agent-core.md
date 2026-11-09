# WP1·2·5 핵심 구현 및 인계

기준: backend main `fb759c3`, econmind-docs WP0 계약 `bd97edb`. 구현 전 API·DB 후속 계약 커밋은 docs `cc779b7`임.

## 기본 동작과 의존

기본은 `V2_MULTI_AGENT=false`, `V2_CLAIM_CHECK=false`, `V2_REVISE_MAX_ROUNDS=0`임. 기존 A의 generation()과 공개 응답 필드는 유지함. B는 true/false/2, C는 true/true/2, B-prime은 false/false/2, C-prime은 false/true/2임.

활성화 전에 김건의 LLM purpose/max_output_tokens 인자 지원(WP4), claims.py(WP12), revise.py(WP3)을 필요한 구성에 맞춰 병합해야 함. 미충족이면 본문 추출·모델 호출 전에 작업 error.code=MULTI_AGENT_DEPENDENCY_UNAVAILABLE, retryable=false로 종료함. 이를 A로 자동 하향하지 않음. 지금 main에서 누락된 선행 모듈을 이번 PR이 구현했다고 주장하지 않음.

## 실행과 안전 경계

- collect()는 작업 간 기록을 분리하고 병렬 자식의 순서를 공유함. 기록은128건, 요약120자로 제한함. 전략 기록은 Strategy.agent_trace에 별도로 저장함.
- 강세·레드팀 약세가 병렬로 같은 본문을 읽고 중재가 판정함. 한쪽 응답 실패는 빈 논거, 양쪽 실패는 기존 분석으로 대체함. 잘못된 중재 기본 필드나 기사 인용은 실패로 처리함.
- 생성 콜백은 작업 취소·리스 → 예산 → LLM 순서를 지킴. 정책 거부·취소·리스 예외를 모의 성공으로 바꾸지 않음. 실패한 병렬 호출의 자식 작업을 회수함.
- USE_CRITIC=false이면 주장 검증과 수정의 모델 호출을 하지 않고 unavailable을 유지함. 원문 문장 존재와 의미상 지지는 별도임.
- claims 생성 필드는 방어적으로 파싱하고 주장12·종목6·연결3 상한을 적용함. B는 claims/claim_ids를 공개하지 않음. claims 문장200자만 공개하며 used_evidence.input_excerpt는 계속 비공개임.
- 수정 모듈이 전략 대상 이름·코드·순서를 바꾸거나 지지 주장을 훼손하면 저장 전 거부함. verify 콜백은 async verify(candidate) -> (candidate, verification)이며, revise_once가 후보 claims를 반영할 수 있도록 전체 후보를 반환함.

## 계측·예산·캐시

새로 만드는 usage_event 로거 없이 기존 telemetry.py를 재사용함. bull/bear/judge/revise 단계명이 기존 job_id와 연결됨. 실행 기록의 토큰 값은 실제 전달값이 있을 때만 포함함.

bull/bear/judge/revise 연산을 각 호출 전에 예약함. 정상 최대 비용은 본문10+RAG1+토론3+critic1+수정/재검증4+전략2=21크레딧임. 같은 job 재시도는 누적함. 이는 실제 금액·SDK 내부 재시도/공급자 fallback 횟수와 다름. 기본 job32/day1000을 바꾸지 않음.

새 구성이 켜졌을 때만 generation.pipeline에 플래그·프롬프트 버전·출력 상한·대체 분석 버전을 추가함. 전략 입력의 verification은 status/notes만 전달함.

## 팀 전달 자료

- 프론트 모의 응답: [합성 JSON](examples/multi-agent-response.json). 실제 성능·실연동 결과가 아님.
- stage: debating / judging / revising. 프론트가 먼저 또는 함께 대응해야 함.
- 배포 필요 + 새 설정 키: V2_MULTI_AGENT=false, V2_CLAIM_CHECK=false, V2_REVISE_MAX_ROUNDS=0.
- 김건 담당: 선행 모듈 병합, Compose 전달, 실제 저장·응답 잘림·계측·예산 스모크 후 활성화함.
- 비교 도구: [사용법 및 재현 경계](compare-multi-agent.md). fixture 재생을 실데이터 재생성으로 기록하지 않음.

## 발표에서 설명할 결정

1. 반대 입장을 병렬로 수집하고 중재가 근거를 비교하게 하여 긍정 편향을 측정 가능한 대상으로 삼음. 편향이 줄었다는 결과는 아직 없음.
2. 프레임워크 도입 없이 기존 큐·리스·멱등성 구조 안에 함수로 연결함.
3. 계약을 먼저 커밋하고 독립 플래그로 구성별 비교와 기존 경로 보존을 검증함.

운영 배포·운영 DB·실제 모델 호출·실데이터 A/B/C 재생성·사람 블라인드 채점은 미실행함.
