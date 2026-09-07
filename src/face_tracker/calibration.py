"""Pinhole intrinsics; reject aspect/crop mismatch rather than pretend calibrated."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import CameraIntrinsics


@dataclass
class CameraModel:
    intrinsics: CameraIntrinsics
    distortion: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        p = self.intrinsics
        return np.array([[p.fx, 0, p.cx], [0, p.fy, p.cy], [0, 0, 1]], dtype=np.float64)

    def undistort(self, points: np.ndarray) -> np.ndarray:
        return cv2.undistortPoints(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
                                   self.matrix, self.distortion, P=self.matrix).reshape(-1, 2)


class CalibrationProvider:
    def __init__(self, hfov: float, path: Path | None = None):
        self.hfov = hfov
        self.data = json.loads(path.read_text(encoding='utf-8-sig')) if path else None
        self.cache: dict[tuple[int, int], CameraModel] = {}

    def at(self, width: int, height: int) -> CameraModel:
        if (width, height) in self.cache:
            return self.cache[width, height]
        if self.data is None:
            camera = CameraModel(CameraIntrinsics.from_horizontal_fov(width, height, self.hfov), np.zeros(5))
        else:
            d = self.data
            if d.get('example_only'):
                raise ValueError('Example calibration is not measured; replace the values and remove example_only')
            if d.get('model', 'pinhole') != 'pinhole':
                raise ValueError('Only OpenCV pinhole calibration is supported, not fisheye')
            values = [float(d[k]) for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy')]
            if not all(math.isfinite(v) for v in values) or min(values[:4]) <= 0:
                raise ValueError('Invalid camera calibration')
            w, h, fx, fy, cx, cy = values
            if width <= 0 or height <= 0 or abs((width / height) / (w / h) - 1) > 0.001:
                raise ValueError('Calibration aspect ratio differs from capture; recalibrate the camera crop')
            distortion = np.asarray(d.get('distortion', [0] * 5), dtype=np.float64).reshape(-1)
            if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
                raise ValueError('Invalid OpenCV distortion coefficients')
            camera = CameraModel(CameraIntrinsics(fx * width / w, fy * height / h,
                cx * width / w, cy * height / h, calibrated=True), distortion)
        self.cache[width, height] = camera
        return camera
