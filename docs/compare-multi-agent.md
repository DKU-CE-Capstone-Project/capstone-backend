# WP8 고정 근거 비교 도구

기준 계약은 econmind-docs `docs/28-contract-wp0-20261108.md`(커밋 `bd97edb`)임.
`scripts/compare_ab.py`는 로컬 파일의 동일한 `used_evidence`와 `rag_evidence`로
구성별 결과를 비교하고, 사람 평가용 표를 만듦. 운영 DB에 연결하거나 Diffbot·NAVER·RAG
검색을 다시 실행하지 않음. 운영 작업 예산 문맥에도 들어가지 않음.

## 구성

| 이름 | 토론 | 주장 검증 | 수정 최대 라운드 | 생성 진입 호출 상한 |
|---|---|---|---|---|
| A | off | off | 0 | 4 |
| B | on | off | 2 | 10 |
| C | on | on | 2 | 10 |
| B-prime(B′) | off | off | 2 | 8 |
| C-prime(C′) | off | on | 2 | 8 |

- 위 상한은 분석·검증·수정과 전략의 응답 검증 재시도 1회를 포함함. 본문 추출·RAG
  조회는 이미 고정된 입력이므로 제외함. `llm.generate` 또는 `generate_with_usage`의
  최상위 진입을 세며, SDK 내부 재시도·공급자 fallback HTTP 횟수·금액 상한은 아님.
- D(조회 도구)는 계약 범위 밖이므로 실행 전에 거부함.
- B/C/B′/C′의 필요한 모듈 또는 호출별 출력 상한 인자가 없으면 선택한 전체 구성을
  모델 호출 전에 거부함. 토론만 실행하고 이를 B 전체라고 표시하지 않음.
- 실제 키·호출 계획은 김건과 맞춘 뒤 별도 실험 환경에서 live 모드를 실행해야 함.

## 입력

다음과 같은 영속화 리포트의 로컬 JSON export를 받음. 원문 발췌 `input_excerpt`는
반드시 있어야 함. 공개 `/api/v2/reports/{id}`와 WP7의 공개 응답 JSON에는 이 필드가
없으므로 그대로 재생할 수 없음. 이 도구 자체에는 DB 읽기 기능이 없음.

```json
{
  "report": {
    "used_evidence": [
      {"news_id": "news-001", "input_excerpt": "[합성 예시] 기업의 수출이 증가했다.", "title": "수출 보도"}
    ],
    "rag_evidence": []
  },
  "strategy_options": {"risk_level": "medium", "period": "short"}
}
```

영속화 문서의 근거 메타데이터도 함께 보존해서 export하는 것을 권장함.
선택 기사 1~10건, 기사당 12,000자·전체 60,000자, RAG 3건·설명 4,000자 상한을 검사함.
원본 입력은 복사해서 각 구성에 전달하고 입력 digest를 기록함. 비밀 설정이나 서버
환경 파일을 export에 넣지 않으며, 입력 JSON과 평가 자료의 보관·공유 범위를 관리함.

## 외부 호출 없는 실행

`--fixtures`와 `--live` 중 하나를 명시해야 함. fixture는 이미 준비한 출력의 재생이며,
모델 품질·모델 지연의 새 실측으로 해석하지 않음. fixture 경로는 앱 설정 모듈이나
공급자 SDK를 import하지 않음.

`fixtures.json`은 다음 형식임. 각 구성의 `report`에는 기존 6개 필수 필드
(`title`, `event_summary`, `event_analysis`, `market_impact`, `stock_impacts`, `risk_factors`),
`verification`에는 `status`·`notes`, `strategy`에는 `strategy_items`를 넣음.

```json
{
  "arms": {
    "A": {
      "report": {"title": "합성 결과", "event_summary": "수출 증가", "event_analysis": "추가 확인 필요", "market_impact": "판단 유보", "stock_impacts": [], "risk_factors": []},
      "verification": {"status": "unavailable", "notes": ["합성 fixture"]},
      "strategy": {"strategy_summary": "관망", "risk_warning": "합성 예시", "strategy_items": []}
    }
  }
}
```

```sh
python -m scripts.compare_ab --input evidence-export.json --fixtures fixtures.json --arms A --output-dir comparison-fixture --seed 19
```

## 별도 환경의 실제 생성

```sh
python -m scripts.compare_ab --input evidence-export.json --live --arms A B C --max-calls 24 --output-dir comparison-live --seed 19
```

위 명령은 API를 호출하는 예시임. 필요한 WP4·WP3·WP12 의존성이 없으면 고급 구성을 거부함. A/B/C 1회씩의 보수적 생성 진입 상한은 24회임.
`--repeats 2`이면 최소 `--max-calls 48`이 필요함. 회차는 1~20회로 제한함.

live 실행은 독립 CLI 프로세스 안에서 구성 플래그와 생성 함수·telemetry 수집기를
일시 교체하고 종료 시 복원함. API 서버·워커 프로세스 안에서 병렬 요청과 함께
호출하지 않음. CLI는 구성들을 순서대로 실행하고, 토론 내부의 병렬 생성은 유지함.

## 여러 근거 묶음을 한 번에 비교

`--input`을 반복해서 최대 10개까지 지정하면 각 묶음을 같은 A/B/C 구성으로 실행함.
같은 입력을 여러 번 지정하면 거부함. 동일 근거의 반복 생성에는 `--repeats`를 사용함.
모든 입력과 전체 계획 상한을 모델 호출 전에
검사하고 하나의 `--max-calls`를 공유함. A/B/C 기준 최대 계획은 묶음 수 × 반복 수 ×
24회임. 예를 들어 2개 묶음을 1회씩 비교하는 live 명령은 다음과 같음.

```sh
python -m scripts.compare_ab --input evidence-01.json --input evidence-02.json --live --arms A B C --max-calls 48 --output-dir comparison-batch --seed 19
```

- 여러 입력의 산출물은 스키마 2임. 입력 순서대로 `case-001`부터 부여하며 `comparison.json`의
  `cases`에 묶음별 근거 해시와 기사 ID를 저장함. 단일 입력의 스키마와 CSV 열은 유지함.
- 전체 완료 결과를 섞어 블라인드 CSV 한 장으로 출력함. `case_id`로 해당 근거 묶음을
  찾을 수 있고, 평가 ID는 실행마다 새로 부여함. 채점자에게는 구성과 반복 회차를 숨긴
  채점표 및 `case_id`에 대응하는 근거를 전달함.
- 다중 입력 fixture 모드는 같은 구성별 출력 fixture를 재사용함. 채점표 형식과 실행
  순서를 확인하는 용도임. 여러 묶음의 품질 차이는 실제 재생성 뒤에 평가함.

## 산출물과 평가

- `comparison.json`: 입력 digest, 구성·반복 회차, 실행 모드, 결과·검증, 실제 경과 시간,
  생성 진입 횟수, 관측된 토큰 정보 등을 저장함. 원본 기사 발췌와 관련 없는 입력
  메타데이터는 출력하지 않음. 실패 원문 대신 고정 오류 코드만 기록함.
- `comparison.csv`: 구성별 경고 수, 상승 방향 수/종목 영향 수, 매수 수/전략 항목 수,
  지연·사용량을 기록함. 관측되지 않은 토큰·비용은 빈 값이며 0으로 만들지 않음.
  금액 계산은 이 도구에서 하지 않으므로 `est_cost_usd`는 live에서도 null임.
- `blind_scores.csv`: seed로 섞은 공통 보고서·전략 내용과 빈 0/1/2점 칸을 출력함.
  구성·회차·trace·검증 경고·claims 등의 구조 식별자는 숨김. 내용 자체에서 방법이
  추측될 가능성까지 제거하는 완전한 눈가림은 아님. 근거 채점자는 원본 입력을 별도로
  읽어야 하며, 근거 인용 정확도와 중요 정보 보존을 구분해서 채점함.
- `answer_key.json`: 평가 ID와 구성·반복 회차를 연결함. 채점자에게는 이 파일과
  `comparison.*`를 함께 공유하지 않고, 채점이 끝난 뒤 집계할 때 사용함.

출력 디렉터리는 새로 만들어야 하며 기존 결과를 덮어쓰지 않음. fixture의 경과 시간은
파일 재생 실행 시간이고 실제 모델 지연이 아님. 자동으로 사람 점수를 채우지 않음.
이번 검증은 가짜 공급자 응답·fixture CLI·오류·상한·눈가림·입력 분리 테스트임.
실데이터 A/B/C 재생성과 사람 채점 30건은 별도 후속 작업으로 남음.
