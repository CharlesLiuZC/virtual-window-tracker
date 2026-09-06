"""Spatial continuity for one viewer; this does not identify a person."""

from math import hypot


class FaceSelector:
    def __init__(self, hold_seconds: float = 0.7) -> None:
        self.hold_seconds = hold_seconds
        self._box: tuple[float, float, float, float] | None = None
        self._seen_at = 0.0

    def select(self, boxes: list[tuple[float, float, float, float]], now: float) -> int | None:
        if not boxes:
            return None
        selected = None
        if self._box is not None and now - self._seen_at < self.hold_seconds:
            x, y, w, h = self._box
            distances = [hypot(b[0] + b[2] / 2 - x - w / 2,
                               b[1] + b[3] / 2 - y - h / 2) for b in boxes]
            nearest = min(range(len(boxes)), key=distances.__getitem__)
            candidate = boxes[nearest]
            ratio = candidate[2] * candidate[3] / max(w * h, 1e-9)
            if distances[nearest] <= max(w, h) * 0.6 and 0.4 <= ratio <= 2.5:
                selected = nearest
            else:
                return None
        if selected is None:
            selected = max(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3])
        self._box = boxes[selected]
        self._seen_at = now
        return selected
