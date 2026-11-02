# NATS 준비 상태 검사 (2026-11-02)

팀 위키의 10-26 회의록은 진행 내역을 저장소에 남겨 발표 자료 취합에 쓰도록 요청한다.
문주안의 10-25 개발·운영 기록은 worker running과 API healthy를 구분한다.
이 후속은 M5 잔여인 NATS 준비 검사를 추가한다. 큐 장애를 주입한 모의 회귀에서
기존 옵션에서는 /ready 200, NATS_REQUIRED=true에서는 503인지 검증한다.

- 새 설정 `NATS_REQUIRED=false`는 기존 판정을 유지하고 `nats: off`를 반환한다.
- true이면 v2 reports_v2 stream 조회·subject·WORK_QUEUE·유한 TTL을 검사한다.
  검사 2초, 연결 정리 최대 0.5초로 제한하며 성공 on/실패 unavailable과 503을 반환한다.
- 새 임시 연결을 닫고 재연결하지 않는다. 큐·소비자를 생성하거나 작업을 발행하지 않는다.
- worker 생존·소비·공급자 성공까지 보증하는 검사는 아니다.
- 배포 필요. 김건이 기존 v2 큐와 조회 권한을 확인한 후 활성화한다.
  GCP 배포·환경 열람·운영 DB 접근·실제 공급자 호출은 이번 작업에서 미실행이다.

근거: [10-26 회의록](https://app.notion.com/p/40514a4e8feb819a8f96edaec9e7fc01),
[개발·운영](https://app.notion.com/p/3da14a4e8feb81a883d3c8d4d9e9a898).
