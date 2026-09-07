"""Temporal association and outlier rejection, independent of camera/MediaPipe."""
from __future__ import annotations

import math

from .filtering import PositionFilter, RestAwarePositionFilter

Position = tuple[float, float, float]


class StablePositionTracker:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 4.0,
                 derivative_cutoff: float = 1.0, hold_seconds: float = 0.75,
                 max_speed_m_s: float = 1.5, rest_aware: bool = True) -> None:
        filter_type = RestAwarePositionFilter if rest_aware else PositionFilter
        self.filter = filter_type(min_cutoff, beta, derivative_cutoff)
        self.hold_seconds = hold_seconds
        self.max_speed_m_s = max_speed_m_s
        self.previous: Position | None = None
        self.last_valid: float | None = None
        self.track_id = 0

    @property
    def stationary(self) -> bool:
        return bool(getattr(self.filter, 'stationary', False))

    def update(self, candidates: list[Position], timestamp_s: float
               ) -> tuple[int, Position] | None:
        if not math.isfinite(timestamp_s):
            return None
        if self.last_valid is not None and timestamp_s <= self.last_valid:
            return None
        valid = [(i, p) for i, p in enumerate(candidates)
                 if all(math.isfinite(v) for v in p) and 0.15 <= p[2] <= 3.0]
        if not valid:
            return None  # Preserve filter/identity through a short occlusion.
        expired = self.last_valid is None or timestamp_s - self.last_valid > self.hold_seconds
        if expired:
            selected = valid[0]  # Caller ranks initial candidates by face area.
            self.filter.reset()
            self.track_id += 1
        else:
            assert self.previous is not None and self.last_valid is not None
            selected = min(valid, key=lambda item: math.dist(item[1], self.previous))
            # Bounded displacement: reject a one-frame detection/eye-distance spike.
            allowance = 0.025 + self.max_speed_m_s * min(timestamp_s - self.last_valid, 0.15)
            if math.dist(selected[1], self.previous) > allowance:
                return None
        index, position = selected
        filtered = self.filter.apply(*position, timestamp_s)
        self.previous, self.last_valid = position, timestamp_s
        return index, filtered
