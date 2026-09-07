"""Same-frame local A/B diagnostic. Stop the service before opening its camera.

No images or landmark recordings are written. Only aggregate position changes
are printed; these are not ground-truth motion or biometric accuracy measures.
"""
import argparse
import json
import time
from dataclasses import replace

import cv2
import mediapipe as mp
import numpy as np

from benchmark_jitter import previous_pose
from face_tracker.config import Settings
from face_tracker.head_pose import LANDMARK_IDS
from face_tracker.service import TrackingService
from face_tracker.stabilization import StablePositionTracker
from face_tracker.tracker_factory import create_tracker


def summary(values):
    a = np.asarray(values)
    if a.size == 0:
        return {'valid': 0, 'step_p95_mm': None, 'step_max_mm': None}
    valid = np.isfinite(a).all(axis=1)
    adjacent = valid[:-1] & valid[1:]
    steps = np.linalg.norm(np.diff(a, axis=0)[adjacent], axis=1) * 1000
    return {'valid': int(valid.sum()),
            'step_p95_mm': round(float(np.percentile(steps, 95)), 3) if steps.size else None,
            'step_max_mm': round(float(steps.max()), 3) if steps.size else None}


def run(seconds):
    settings = replace(Settings.from_env(), tracker_backend='landmarker')
    capture = TrackingService(settings)._open_camera()
    series = {k: [] for k in ['previous_raw', 'new_raw', 'previous_filtered', 'new_filtered']}
    old_filter = StablePositionTracker(rest_aware=False)
    counts = {'frames': 0, 'no_face': 0, 'stationary': 0, 'continuous': 0, 'reinitialize': 0}
    solver_times = []
    try:
        with create_tracker(settings) as tracker:
            started = time.monotonic()
            previous_ms = -1
            while time.monotonic() - started < seconds:
                ok, frame = capture.read()
                if not ok:
                    break
                now = time.monotonic()
                timestamp_ms = max(previous_ms + 1, int(now * 1000))
                previous_ms = timestamp_ms
                result = tracker.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp_ms)
                counts['frames'] += 1
                if not result.face_landmarks:
                    counts['no_face'] += 1
                    for rows in series.values():
                        rows.append([np.nan] * 3)
                    continue
                h, w = frame.shape[:2]
                camera = tracker.camera.at(w, h)
                pixels = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])[LANDMARK_IDS]
                old = previous_pose(tracker.pose.object_points, pixels, camera, w)
                solve_started = time.perf_counter()
                pose = tracker.pose.solve(pixels, camera, w, timestamp_ms / 1000)
                solver_times.append((time.perf_counter() - solve_started) * 1000)
                new = pose.position if pose else None
                if pose:
                    counts[pose.solver] += 1
                a = old_filter.update([tuple(old)] if old is not None else [], now)
                b = tracker.stable.update([new] if new is not None else [], now)
                counts['stationary'] += bool(b and tracker.stable.stationary)
                for key, value in [('previous_raw', old), ('new_raw', new),
                                   ('previous_filtered', a[1] if a else None),
                                   ('new_filtered', b[1] if b else None)]:
                    series[key].append(value if value is not None else [np.nan] * 3)
    finally:
        capture.release()
    return {'note': 'Same-frame A/B; uncontrolled real motion, no ground truth. No frames or landmark history saved.',
            'counts': counts, 'series': {k: summary(v) for k, v in series.items()},
            'new_pose_solver_ms_mean': round(float(np.mean(solver_times)), 3) if solver_times else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=15)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 60:
        parser.error('--seconds must be between 3 and 60')
    print(json.dumps(run(args.seconds), indent=2))
