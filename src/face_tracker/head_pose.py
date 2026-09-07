"""Generic-face PnP, scaled by assumed IPD. Not biometric identity recognition.

Canonical points: Google MediaPipe canonical_face_model.obj (Apache-2.0),
https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/face_geometry/data
Coordinates retain the original vertex indexing; only rigid-ish regions are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CameraModel
from .geometry import euler_degrees_from_rotation_matrix

LANDMARK_IDS = [1, 4, 6, 168, 10, 151, 33, 133, 362, 263, 127, 356, 234, 454, 93, 323]
CANONICAL_POINTS = np.array([
    [0, -1.126865, 7.475604], [0, -.463170, 7.586580], [0, 2.473255, 5.788627],
    [0, 3.271027, 5.236015], [0, 8.261778, 4.481535], [0, 6.545390, 5.027311],
    [-4.445859, 2.663991, 3.173422], [-1.856432, 2.585245, 3.757904],
    [1.856432, 2.585245, 3.757904], [4.445859, 2.663991, 3.173422],
    [-7.743095, 2.364999, -2.005167], [7.743095, 2.364999, -2.005167],
    [-7.664182, .673132, -2.435867], [7.664182, .673132, -2.435867],
    [-7.542244, -1.049282, -2.431321], [7.542244, -1.049282, -2.431321],
], dtype=np.float64)


def face_object_points(ipd_m: float) -> np.ndarray:
    if not np.isfinite(ipd_m) or not 0.03 <= ipd_m <= 0.09:
        raise ValueError('Assumed IPD must be between 0.03 and 0.09 m')
    right = CANONICAL_POINTS[[6, 7]].mean(axis=0)
    left = CANONICAL_POINTS[[8, 9]].mean(axis=0)
    # Origin at eye midpoint; OpenCV x right, y down, z into the head.
    return (CANONICAL_POINTS - (left + right) / 2) * np.array([1, -1, -1]) * ipd_m / np.linalg.norm(left - right)


@dataclass
class HeadPose:
    position: tuple[float, float, float]
    angles: dict[str, float]
    reprojection_px: float
    inlier_ratio: float
    calibration_ready: bool
    solver: str = 'reinitialize'


class HeadPoseEstimator:
    def __init__(self, ipd_m: float):
        self.object_points = face_object_points(ipd_m)
        self.reset()

    def reset(self):
        self._previous = None
        self._timestamp = None
        self._camera_key = None

    def _refine_robust(self, pixels, matrix, distortion, rvec, tvec, scale):
        """Continuous Cauchy weights avoid fitting a different hard subset each frame.

        projectPoints supplies the pose Jacobian (rotation then translation).
        This optimizes this frame's measurements, not a low-pass of old poses.
        """
        parameters = np.concatenate((rvec.ravel(), tvec.ravel())).astype(np.float64)
        damping = 1e-3
        for _ in range(12):
            projected, jacobian = cv2.projectPoints(self.object_points, parameters[:3], parameters[3:], matrix, distortion)
            residual = pixels - projected.reshape(-1, 2)
            squared = np.sum(residual ** 2, axis=1)
            cost = np.log1p(squared / scale ** 2).sum()
            weights = np.repeat(1 / (1 + squared / scale ** 2), 2)
            jacobian = jacobian[:, :6]
            normal = jacobian.T @ (weights[:, None] * jacobian)
            gradient = jacobian.T @ (weights * residual.ravel())
            accepted = False
            for _ in range(5):
                try:
                    step = np.linalg.solve(normal + damping * np.diag(np.maximum(np.diag(normal), 1e-6)), gradient)
                except np.linalg.LinAlgError:
                    return None
                if not np.isfinite(step).all():
                    return None
                candidate = parameters + step
                if not .1 < candidate[5] < 4:
                    damping *= 10
                    continue
                trial, _ = cv2.projectPoints(self.object_points, candidate[:3], candidate[3:], matrix, distortion)
                trial_squared = np.sum((pixels - trial.reshape(-1, 2)) ** 2, axis=1)
                if np.log1p(trial_squared / scale ** 2).sum() <= cost + 1e-12:
                    parameters = candidate
                    damping = max(1e-7, damping / 3)
                    accepted = True
                    break
                damping *= 10
            if not accepted or np.linalg.norm(step) < 1e-8:
                break
        return parameters[:3].reshape(3, 1), parameters[3:].reshape(3, 1)

    def _validate(self, pixels, matrix, distortion, rvec, tvec, threshold, solver):
        projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, matrix, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
        if not np.isfinite(errors).all():
            return None
        # Hard thresholds only validate the result; they no longer change the fit.
        inliers = errors <= threshold
        if np.count_nonzero(inliers) < 12:
            return None
        rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
        rotation, _ = cv2.Rodrigues(rvec)
        position = tvec.ravel()
        if not np.isfinite(position).all() or not 0.15 <= position[2] <= 3:
            return None
        if np.any((self.object_points @ rotation.T + position)[:, 2] <= 0):
            return None
        angles = euler_degrees_from_rotation_matrix(rotation)
        if abs(angles['yaw']) > 65 or abs(angles['pitch']) > 50 or abs(angles['roll']) > 60:
            return None
        return HeadPose((float(position[0]), float(-position[1]), float(position[2])), angles, rms,
                        float(np.mean(inliers)),
                        abs(angles['yaw']) < 20 and abs(angles['pitch']) < 25 and abs(angles['roll']) < 25,
                        solver)

    def solve(self, image_points: np.ndarray, camera: CameraModel, width: int,
              timestamp_s: float | None = None) -> HeadPose | None:
        image_points = np.asarray(image_points, dtype=np.float64)
        if image_points.shape != (len(LANDMARK_IDS), 2) or not np.isfinite(image_points).all():
            return None
        if timestamp_s is not None and (not np.isfinite(timestamp_s)
                or self._timestamp is not None and timestamp_s <= self._timestamp):
            return None
        matrix, distortion = camera.matrix, camera.distortion
        camera_key = (tuple(matrix.ravel()), tuple(distortion.ravel()))
        warm = (timestamp_s is not None and self._timestamp is not None
                and timestamp_s - self._timestamp <= .25 and camera_key == self._camera_key)
        threshold = max(2.0, width * 0.006)
        robust_scale = max(1.0, threshold / 3)
        try:
            for solver in (['continuous', 'reinitialize'] if warm else ['reinitialize']):
                if solver == 'continuous':
                    rvec, tvec = (p.copy() for p in self._previous)
                else:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(self.object_points, image_points,
                        matrix, distortion, iterationsCount=100, reprojectionError=threshold,
                        confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
                    if not ok or inliers is None or len(inliers) < 12:
                        continue
                refined = self._refine_robust(image_points, matrix, distortion, rvec, tvec, robust_scale)
                if refined is None:
                    continue
                rvec, tvec = refined
                pose = self._validate(image_points, matrix, distortion, rvec, tvec, threshold, solver)
                if pose is None:
                    continue
                self._previous = (rvec.copy(), tvec.copy())
                self._timestamp = timestamp_s
                self._camera_key = camera_key
                return pose
        except cv2.error:
            pass
        # Do not use a rejected solution as next frame's initial estimate.
        self.reset()
        return None
