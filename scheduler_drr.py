# scheduler_drr.py
import time
import asyncio
import collections
import contextlib
from dataclasses import dataclass, field
from typing import Deque, Dict, Callable, Awaitable, Any

@dataclass
class DocTask:
    req_id: int
    doc_meta: Any          # pb.EmbedRequest.DocumentMeta
    webhook_info: Any      # pb.WebhookInfo
    enqueue_ts: float = field(default_factory=lambda: time.monotonic())

@dataclass
class TenantQ:
    req_id: int
    weight: int = 5        # 내부 가중치(1~10 정도 권장)
    deficit: int = 0
    q: Deque[DocTask] = field(default_factory=collections.deque)
    last_served_ts: float = field(default_factory=lambda: time.monotonic())


class DRRScheduler:
    """
    Deficit Round Robin 스케줄러.
    - enqueue(req_id, DocTask)
    - start(worker_coro): 라운드 돌며 크레딧 배분, 작업 실행
    - set_weight(req_id, weight): 필요시 내부 가중치 조정(현재 RPC는 없으므로 내부전용)
    """
    def __init__(self, quantum: int = 1, aging_sec: float = 8.0, aging_bonus: int = 1):
        self.quantum = quantum
        self.aging_sec = aging_sec
        self.aging_bonus = aging_bonus
        self.tenants: Dict[int, TenantQ] = {}
        self._wakeup = asyncio.Event()
        self._running = False

    def enqueue(self, req_id: int, task: DocTask):
        tq = self.tenants.setdefault(req_id, TenantQ(req_id=req_id))
        tq.q.append(task)
        self._wakeup.set()

    def set_weight(self, req_id: int, weight: int):
        tq = self.tenants.setdefault(req_id, TenantQ(req_id=req_id))
        tq.weight = max(1, min(10, weight))
        self._wakeup.set()

    async def start(self, worker_coro: Callable[[DocTask], Awaitable[None]]):
        self._running = True
        while self._running:
            made = False
            now = time.monotonic()

            # 에이징: 오래 못 돌면 보너스 크레딧
            for tq in self.tenants.values():
                if (now - tq.last_served_ts) >= self.aging_sec:
                    tq.deficit += self.aging_bonus

            for req_id, tq in list(self.tenants.items()):
                if not tq.q:
                    continue
                # 라운드 크레딧 지급(가중치 반영)
                tq.deficit += self.quantum * tq.weight

                # 문서 하나당 코스트=1 (필요시 크기/페이지 기반 가중치로 확장 가능)
                while tq.q and tq.deficit >= 1:
                    job = tq.q.popleft()
                    tq.deficit -= 1
                    tq.last_served_ts = time.monotonic()
                    made = True
                    # 워커는 독립 태스크로 수행
                    asyncio.create_task(worker_coro(job))

                # 과도한 크레딧 누적 방지(옵션)
                if not tq.q and tq.deficit > 10:
                    tq.deficit = 10

            if not made:
                self._wakeup.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wakeup.wait(), timeout=0.2)

    def stop(self):
        self._running = False
        self._wakeup.set()