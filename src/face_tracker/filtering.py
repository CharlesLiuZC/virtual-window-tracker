from __future__ import annotations

import math
from collections import deque
from statistics import median


class LowPassFilter:
    def __init__(self) -> None:
        self._value: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


class OneEuroFilter:
    """Adaptive low-pass filter with low jitter and limited motion lag."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.035,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0 or derivative_cutoff <= 0:
            raise ValueError("Filter cutoff frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._signal = LowPassFilter()
        self._derivative = LowPassFilter()
        self._previous_raw: float | None = None
        self._previous_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value: float, timestamp_s: float) -> float:
        if self._previous_time is None or timestamp_s <= self._previous_time:
            self._previous_raw = value
            self._previous_time = timestamp_s
            return self._signal.apply(value, 1.0)

        dt = max(timestamp_s - self._previous_time, 1e-6)
        derivative = (value - self._previous_raw) / dt
        filtered_derivative = self._derivative.apply(
            derivative, self._alpha(self.derivative_cutoff, dt)
        )
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        filtered = self._signal.apply(value, self._alpha(cutoff, dt))
        self._previous_raw = value
        self._previous_time = timestamp_s
        return filtered

    def reset(self) -> None:
        self._signal.reset()
        self._derivative.reset()
        self._previous_raw = None
        self._previous_time = None


class PositionFilter:
    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float) -> None:
        args = (min_cutoff, beta, derivative_cutoff)
        self.x = OneEuroFilter(*args)
        self.y = OneEuroFilter(*args)
        self.z = OneEuroFilter(*args)

    def apply(
        self, x: float, y: float, z: float, timestamp_s: float
    ) -> tuple[float, float, float]:
        return (
            self.x.apply(x, timestamp_s),
            self.y.apply(y, timestamp_s),
            self.z.apply(z, timestamp_s),
        )

    def reset(self) -> None:
        self.x.reset()
        self.y.reset()
        self.z.reset()


class RestAwarePositionFilter:
    """Bounded static noise region with hysteresis, not a frozen camera.

    Confirm rest from 450 ms of stable raw observations. One isolated excursion
    cannot release the anchor; two consecutive, same-direction observations can.
    Slow deliberate motion eventually leaves the region and is never accumulated
    against a moving noise reference. All distances use the estimator's scale,
    which is still assumed-IPD, not metrologically calibrated.
    """
    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float,
                 xy_radius_m: float = .003, z_radius_m: float = .006):
        self.base = PositionFilter(min_cutoff, beta, derivative_cutoff)
        self.base.z = OneEuroFilter(min_cutoff * .7, beta * .6, derivative_cutoff)
        self.radii = (xy_radius_m, xy_radius_m, z_radius_m)
        self.reset()

    def reset(self):
        self.base.reset()
        self.history = deque()
        self.anchor = None
        self.center = None
        self.output = None
        self.previous_time = None
        self.departure = None

    @property
    def stationary(self):
        return self.anchor is not None

    def apply(self, x, y, z, timestamp_s):
        position = (x, y, z)
        dt = timestamp_s - self.previous_time if self.previous_time is not None else 0
        self.previous_time = timestamp_s
        if dt > .25:
            # Never infer continuous stillness across an unobserved interval.
            self.history.clear()
            self.anchor = self.center = self.departure = None
        if self.anchor is not None:
            displacement = tuple((p - c) / r for p, c, r in zip(position, self.center, self.radii))
            outside = math.sqrt(sum(d * d for d in displacement)) > 1.6
            confirmed = (outside and self.departure is not None
                         and sum(a * b for a, b in zip(displacement, self.departure)) > 0)
            self.departure = displacement if outside else None
            if not confirmed:
                return self.anchor
            # Warm the dynamic filter from the held output at the current frame
            # interval. Otherwise the long stationary dt would create a snap.
            self.base.reset()
            self.base.apply(*self.anchor, timestamp_s - max(.001, min(dt, .1)))
            self.anchor = self.center = self.departure = None
            self.history.clear()

        self.output = self.base.apply(*position, timestamp_s)
        self.history.append((timestamp_s, position))
        while self.history and timestamp_s - self.history[0][0] > .6:
            self.history.popleft()
        if len(self.history) >= 4 and timestamp_s - self.history[0][0] >= .45:
            center = tuple(median(p[axis] for _, p in self.history) for axis in range(3))
            if all(sum(((p[i] - center[i]) / self.radii[i]) ** 2 for i in range(3)) <= 1
                   for _, p in self.history):
                self.center = center
                self.anchor = self.output
                self.departure = None
        return self.output
