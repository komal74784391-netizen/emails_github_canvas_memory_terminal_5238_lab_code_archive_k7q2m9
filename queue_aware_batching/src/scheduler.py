"""Queue-aware batching scheduler."""

from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    request_id: str
    prompt_tokens: int
    max_tokens: int
    arrival_ms: float


class QueueAwareScheduler:
    def __init__(self, max_batch: int = 32, latency_budget_ms: float = 250.0):
        self.max_batch = max_batch
        self.latency_budget_ms = latency_budget_ms

    def _urgency(self, request: Request) -> float:
        # Requests with longer forecasts and older arrivals are more urgent.
        return request.prompt_tokens + request.max_tokens + request.arrival_ms * 0.01

    def select_batch(self, queue: List[Request], now_ms: float) -> List[Request]:
        ranked = sorted(queue, key=self._urgency, reverse=True)
        batch: List[Request] = []
        token_budget = 0
        for req in ranked:
            estimate = req.prompt_tokens + req.max_tokens
            if token_budget + estimate > self.max_batch * 512:
                continue
            batch.append(req)
            token_budget += estimate
            if len(batch) >= self.max_batch:
                break
        return batch
