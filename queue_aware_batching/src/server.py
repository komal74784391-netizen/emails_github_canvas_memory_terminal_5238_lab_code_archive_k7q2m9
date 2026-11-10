"""Minimal reference inference server."""

from typing import Dict, List

from .scheduler import QueueAwareScheduler, Request


class ReferenceServer:
    def __init__(self):
        self.scheduler = QueueAwareScheduler()
        self.queue: List[Request] = []

    def enqueue(self, request: Request) -> None:
        self.queue.append(request)

    def step(self, now_ms: float) -> Dict[str, List[str]]:
        batch = self.scheduler.select_batch(self.queue, now_ms)
        ids = [r.request_id for r in batch]
        self.queue = [r for r in self.queue if r.request_id not in ids]
        return {"served": ids}
