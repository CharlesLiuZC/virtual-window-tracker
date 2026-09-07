"""Synthetic yaw sweep, same canonical face and known camera. No live images."""
import json
import math

import cv2
import numpy as np

from face_tracker.calibration import CalibrationProvider
from face_tracker.geometry import Point2, estimate_viewer_position_m
from face_tracker.head_pose import HeadPoseEstimator

camera = CalibrationProvider(70).at(1280, 720)
estimator = HeadPoseEstimator(.063)
rows = []
for yaw in [-45, -30, 0, 30, 45]:
    p, _ = cv2.projectPoints(estimator.object_points, np.array([0., math.radians(yaw), 0.]),
                            np.array([0., 0., .65]), camera.matrix, camera.distortion)
    p = p.reshape(-1, 2)
    a, b = p[[6, 7]].mean(axis=0), p[[8, 9]].mean(axis=0)
    old = estimate_viewer_position_m(Point2(*((a + b) / 2)), np.linalg.norm(a - b), camera.intrinsics, .063)
    pose = estimator.solve(p, camera, 1280)
    rows.append(dict(yaw_deg=yaw, true_z_m=.65, eye_distance_z_m=round(old[2], 4),
                     pnp_z_m=round(pose.position[2], 4) if pose else None))
print(json.dumps({'note': 'Ideal canonical face and correct camera; does NOT validate real-person accuracy.',
                  'rows': rows}, indent=2))
