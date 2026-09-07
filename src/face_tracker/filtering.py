from __future__ import annotations

import math


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
        # Lateral motion drives parallax directly and must not trail the head.
        # Depth remains more strongly damped because eye-scale estimates are noisy.
        args = (min_cutoff, beta * 2.0, derivative_cutoff)
        self.x = OneEuroFilter(*args)
        self.y = OneEuroFilter(*args)
        # Depth from apparent eye spacing is noisier than lateral position.
        self.z = OneEuroFilter(min_cutoff * 0.5, beta * 0.5, derivative_cutoff)

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
    """Freeze sub-centimeter sensor noise without making real motion sticky.

    Entering rest uses a short dwell window. Leaving rest requires two
    directionally consistent samples, so a single landmark/PnP spike cannot
    move the virtual window. The rest reference remains fixed, which prevents
    slow deliberate motion from being absorbed as drift.
    """

    def __init__(
        self,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
        *,
        enter_lateral_m: float = 0.0025,
        enter_depth_m: float = 0.006,
        exit_lateral_m: float = 0.01,
        exit_depth_m: float = 0.02,
        settle_samples: int = 6,
        release_samples: int = 2,
    ) -> None:
        self.filter = PositionFilter(min_cutoff, beta, derivative_cutoff)
        self.enter_lateral_m = enter_lateral_m
        self.enter_depth_m = enter_depth_m
        self.exit_lateral_m = exit_lateral_m
        self.exit_depth_m = exit_depth_m
        self.settle_samples = max(1, settle_samples)
        self.release_samples = max(1, release_samples)
        self._stationary = False
        self._output: tuple[float, float, float] | None = None
        self._rest_reference: tuple[float, float, float] | None = None
        self._rest_count = 0
        self._release_delta: tuple[float, float, float] | None = None
        self._release_count = 0
        self._last_timestamp: float | None = None

    @staticmethod
    def _delta(
        value: tuple[float, float, float],
        reference: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(value, reference))

    @staticmethod
    def _same_direction(
        current: tuple[float, float, float],
        previous: tuple[float, float, float],
    ) -> bool:
        return sum(a * b for a, b in zip(current, previous)) > 0

    def _inside_enter_band(self, delta: tuple[float, float, float]) -> bool:
        return (
            math.hypot(delta[0], delta[1]) <= self.enter_lateral_m
            and abs(delta[2]) <= self.enter_depth_m
        )

    def _outside_exit_band(self, delta: tuple[float, float, float]) -> bool:
        return (
            math.hypot(delta[0], delta[1]) >= self.exit_lateral_m
            or abs(delta[2]) >= self.exit_depth_m
        )

    def _filter_has_settled(
        self,
        raw: tuple[float, float, float],
        filtered: tuple[float, float, float],
    ) -> bool:
        error = self._delta(raw, filtered)
        return math.hypot(error[0], error[1]) <= 0.00075 and abs(error[2]) <= 0.002

    def apply(
        self, x: float, y: float, z: float, timestamp_s: float
    ) -> tuple[float, float, float]:
        raw = (x, y, z)
        gap = (
            timestamp_s - self._last_timestamp
            if self._last_timestamp is not None
            else 0.0
        )
        self._last_timestamp = timestamp_s
        if self._output is None:
            self._output = self.filter.apply(*raw, timestamp_s)
            self._rest_reference = raw
            self._rest_count = 1
            return self._output

        assert self._rest_reference is not None
        delta = self._delta(raw, self._rest_reference)

        if self._stationary:
            # After a capture gap the old anchor is stale; let the underlying
            # filter move immediately instead of requiring confirmation frames.
            if gap > 0.25:
                self._stationary = False
                self._rest_reference = raw
                self._rest_count = 1
                self._release_delta = None
                self._release_count = 0
                self._output = self.filter.apply(*raw, timestamp_s)
                return self._output
            if not self._outside_exit_band(delta):
                self._release_delta = None
                self._release_count = 0
                self.filter.apply(*self._rest_reference, timestamp_s)
                return self._output

            if self._release_delta is not None and self._same_direction(
                delta, self._release_delta
            ):
                self._release_count += 1
            else:
                self._release_count = 1
            self._release_delta = delta

            if self._release_count < self.release_samples:
                self.filter.apply(*self._rest_reference, timestamp_s)
                return self._output

            self._stationary = False
            self._rest_reference = raw
            self._rest_count = 1
            self._release_delta = None
            self._release_count = 0
            self._output = self.filter.apply(*raw, timestamp_s)
            return self._output

        self._output = self.filter.apply(*raw, timestamp_s)
        if self._inside_enter_band(delta) and self._filter_has_settled(
            raw, self._output
        ):
            self._rest_count += 1
            if self._rest_count >= self.settle_samples:
                self._stationary = True
        elif not self._inside_enter_band(delta):
            self._rest_reference = raw
            self._rest_count = 1
        else:
            self._rest_count = 0
        return self._output

    @property
    def output(self) -> tuple[float, float, float] | None:
        return self._output

    @property
    def stationary(self) -> bool:
        return self._stationary

    def reset(self) -> None:
        self.filter.reset()
        self._stationary = False
        self._output = None
        self._rest_reference = None
        self._rest_count = 0
        self._release_delta = None
        self._release_count = 0
        self._last_timestamp = None
