"""Deterministic event ordering for Simulator v2."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .schema import ValidationError


@dataclass(order=True, frozen=True)
class Event:
    timestamp_s: float
    priority: int
    token_id: int
    layer_id: int
    replica_id: str
    sequence_number: int
    kind: str = field(compare=False)
    payload: Any = field(default=None, compare=False)


class EventQueue:
    """Stable priority queue using the ordering contract from the v2 spec."""

    def __init__(self) -> None:
        self._heap: List[Event] = []
        self._next_sequence = 0
        self.now_s = 0.0

    def __len__(self) -> int:
        return len(self._heap)

    def schedule(
        self,
        timestamp_s: float,
        kind: str,
        priority: int = 0,
        token_id: int = 0,
        layer_id: int = 0,
        replica_id: str = "",
        payload: Any = None,
    ) -> Event:
        if not math.isfinite(timestamp_s) or timestamp_s < self.now_s:
            raise ValidationError(
                "event timestamp must be finite and not precede current time"
            )
        if not kind:
            raise ValidationError("event kind must be non-empty")
        event = Event(
            timestamp_s=timestamp_s,
            priority=priority,
            token_id=token_id,
            layer_id=layer_id,
            replica_id=replica_id,
            sequence_number=self._next_sequence,
            kind=kind,
            payload=payload,
        )
        self._next_sequence += 1
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> Event:
        if not self._heap:
            raise IndexError("pop from empty event queue")
        event = heapq.heappop(self._heap)
        if event.timestamp_s < self.now_s:
            raise RuntimeError("event time moved backwards")
        self.now_s = event.timestamp_s
        return event

    def run(
        self,
        handler: Callable[[Event, "EventQueue"], None],
        max_events: Optional[int] = None,
    ) -> int:
        processed = 0
        while self._heap and (max_events is None or processed < max_events):
            event = self.pop()
            handler(event, self)
            processed += 1
        return processed

