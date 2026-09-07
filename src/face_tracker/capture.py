"""Single-slot capture: slow inference skips frames instead of building a queue."""
from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2


class LatestFrameSlot:
    def __init__(self):
        self.condition = threading.Condition()
        self.item = None
        self.dropped = 0

    def put(self, item):
        with self.condition:
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.condition.notify_all()

    def take(self, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.item is not None, timeout)
            item, self.item = self.item, None
            return item

    def clear(self):
        with self.condition:
            self.item = None


@dataclass
class CapturedFrame:
    image: Any
    monotonic_s: float
    unix_ms: int
    epoch: int
    fps: float


class LatestCamera:
    def __init__(self, opener: Callable, stop: threading.Event, status: Callable, *, pace_file: bool = False):
        self.opener, self.stop, self.status = opener, stop, status
        self.pace_file = pace_file
        self.slot = LatestFrameSlot()
        self.epoch = 0
        self.connected = False
        self.thread = threading.Thread(target=self._run, name='camera-latest-frame', daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=3)
        self.slot.clear()

    def is_current(self, sample: CapturedFrame):
        return self.connected and sample.epoch == self.epoch

    def _run(self):
        while not self.stop.is_set():
            capture = None
            try:
                capture = self.opener()
                self.epoch += 1
                self.connected = True
                self.status('running', width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                previous, fps, failures = None, 0.0, 0
                source_fps = capture.get(cv2.CAP_PROP_FPS) if self.pace_file else 0
                period = 1 / source_fps if math.isfinite(source_fps) and source_fps > 0 else 1 / 30
                deadline = time.monotonic()
                while not self.stop.is_set():
                    # Files decode faster than real time; don't drain a replay in one burst.
                    if self.pace_file and self.stop.wait(max(0, deadline - time.monotonic())):
                        break
                    ok, frame = capture.read()
                    if not ok:
                        failures += 1
                        if failures >= 4:
                            raise RuntimeError('Camera stopped returning frames')
                        self.stop.wait(0.05)
                        continue
                    failures = 0
                    now = time.monotonic()
                    deadline = max(deadline + period, now)
                    if previous is not None:
                        instant = 1 / max(now - previous, 1e-6)
                        fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
                    previous = now
                    self.slot.put(CapturedFrame(frame, now, int(time.time() * 1000), self.epoch, fps))
            except Exception as error:
                self.status('reconnecting', f'{type(error).__name__}: {error}')
            finally:
                self.connected = False
                self.slot.clear()
                if capture is not None:
                    capture.release()
            self.stop.wait(1)
