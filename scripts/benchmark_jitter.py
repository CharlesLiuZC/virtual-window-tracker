"""Deterministic geometric regression, not a real-person accuracy benchmark.

Keeps the previous frame-independent PnP fit as an explicit reference. No camera,
images, downloaded assets, or private face data are used.
"""
import json
import math

import cv2
import numpy as np

from face_tracker.calibration import CalibrationProvider
from face_tracker.head_pose import HeadPoseEstimator
from face_tracker.stabilization import StablePositionTracker


def previous_pose(points, pixels, camera, width):
    threshold = max(2., width * .006)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(points, pixels, camera.matrix,
        camera.distortion, iterationsCount=100, reprojectionError=threshold,
        confidence=.99, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 12:
        return None
    ids = inliers.ravel()
    _, tvec = cv2.solvePnPRefineLM(points[ids], pixels[ids], camera.matrix,
                                  camera.distortion, rvec, tvec)
    return tvec.ravel() * [1, -1, 1]


def observations(kind, count=240):
    camera = CalibrationProvider(70).at(1280, 720)
    model = HeadPoseEstimator(.063).object_points
    rng = np.random.default_rng(614)
    # Fixed non-rigid image-space mismatch: a generic head does not match a
    # particular person's face. One point lies near the old hard inlier cutoff.
    bias = np.zeros((16, 2))
    if kind == 'shape_mismatch':
        bias = rng.normal(0, 1.5, (16, 2))
        bias[[4, 10, 13]] += [[5, -4], [-7, 1], [5, 6]]
    for i in range(count):
        x = 0.08 * math.sin((i - 60) / 16) if kind == 'moving' and i >= 60 else 0.
        truth = np.array([x, 0., .65])
        pixels, _ = cv2.projectPoints(model, np.array([.05, .1, 0.]),
                                     truth, camera.matrix, camera.distortion)
        pixels = pixels.reshape(-1, 2) + bias + rng.normal(0, .45, (16, 2))
        yield i / 16, pixels, truth, camera


def stats(positions, truths):
    positions, truths = np.asarray(positions), np.asarray(truths)
    valid = np.isfinite(positions).all(axis=1)
    adjacent = valid[1:] & valid[:-1]
    steps = np.linalg.norm(np.diff(positions, axis=0)[adjacent], axis=1) * 1000
    return {'valid': int(valid.sum()),
            'step_p95_mm': round(float(np.percentile(steps, 95)), 3),
            'step_max_mm': round(float(steps.max()), 3),
            'truth_rmse_mm': round(float(np.sqrt(np.mean(np.sum((positions[valid] - truths[valid]) ** 2, axis=1))) * 1000), 3)}


def run():
    report = {}
    for kind in ['stationary', 'shape_mismatch', 'moving']:
        estimator = HeadPoseEstimator(.063)
        old, new, truths, old_filtered, new_filtered = [], [], [], [], []
        previous_filter = StablePositionTracker(rest_aware=False)
        new_filter = StablePositionTracker()
        stationary_frames = 0
        for time, pixels, truth, camera in observations(kind):
            previous = previous_pose(estimator.object_points, pixels, camera, 1280)
            pose = estimator.solve(pixels, camera, 1280, time)
            old.append(previous if previous is not None else [np.nan] * 3)
            new.append(pose.position if pose else [np.nan] * 3)
            a = previous_filter.update([tuple(previous)] if previous is not None else [], time)
            b = new_filter.update([pose.position] if pose else [], time)
            old_filtered.append(a[1] if a else [np.nan] * 3)
            new_filtered.append(b[1] if b else [np.nan] * 3)
            stationary_frames += new_filter.stationary
            truths.append(truth * [1, -1, 1])
        report[kind] = {'previous_raw': stats(old, truths), 'continuous_robust_raw': stats(new, truths),
                        'previous_filtered': stats(old_filtered[32:], truths[32:]),
                        'new_filtered': stats(new_filtered[32:], truths[32:]),
                        'stationary_frames': stationary_frames}
    return report


if __name__ == '__main__':
    print(json.dumps({'note': 'Synthetic known-pose test; not real-face accuracy or a measured end-to-end latency.',
                      'scenarios': run()}, indent=2))
