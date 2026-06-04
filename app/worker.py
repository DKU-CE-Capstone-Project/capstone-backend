"""
분석 worker 엔트리포인트 (KEDA가 1→N으로 오토스케일하는 대상).

NATS JetStream의 analysis.jobs 주제를 durable pull consumer로 구독하여
한 번에 1개 job을 가져와 run_analysis()를 실행하고 결과를 Redis에 저장한다.

worker 1팟 = 1 in-flight job(동시성 1)로 캡 → replica 수가 곧 병렬도.
이렇게 해야 KEDA 오토스케일 효과를 변수로 분리해 측정할 수 있다.

실행: python -m app.worker
"""
from __future__ import annotations

import asyncio
import json

from app.agents.orchestrator import run_analysis
from app.config import settings
from app.job_store import set_job
from app.messaging.nats_client import DURABLE, SUBJECT, connect, ensure_stream


async def _process(job: dict) -> None:
    """단일 job 처리: run_analysis → Redis 저장."""
    job_id = job.get("job_id", "")
    keyword = job.get("keyword", "")
    try:
        # demo_mode: 고정 처리지연으로 burst 측정의 재현성 확보
        if settings.demo_mode:
            await asyncio.sleep(settings.demo_delay_seconds)

        result = await run_analysis(keyword)
        await set_job(job_id, {
            "status": "done",
            "keyword": keyword,
            "result": json.loads(result.model_dump_json()),
        })
        print(f"[worker] done job={job_id} keyword={keyword}")
    except Exception as exc:  # noqa: BLE001
        await set_job(job_id, {"status": "error", "keyword": keyword, "error": str(exc)})
        print(f"[worker] error job={job_id}: {exc}")


async def main() -> None:
    nc = await connect()
    js = nc.jetstream()
    await ensure_stream(js)
    sub = await js.pull_subscribe(SUBJECT, durable=DURABLE)
    print(f"[worker] subscribed subject={SUBJECT} durable={DURABLE} (demo_mode={settings.demo_mode})")

    while True:
        try:
            msgs = await sub.fetch(1, timeout=5)
        except Exception:
            # fetch timeout — 큐가 비어 있음, 계속 폴링
            continue
        for msg in msgs:
            try:
                job = json.loads(msg.data.decode())
                await _process(job)
                await msg.ack()
            except Exception as exc:  # noqa: BLE001
                print(f"[worker] message handling error: {exc}")
                try:
                    await msg.nak()
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())
