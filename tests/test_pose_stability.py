import cv2
import numpy as np
import pytest

from face_tracker.calibration import CalibrationProvider
from face_tracker.head_pose import HeadPoseEstimator


def project(estimator, camera, x=0., yaw=.1):
    pixels, _ = cv2.projectPoints(estimator.object_points, np.array([0., yaw, 0.]),
                                 np.array([x, 0., .65]), camera.matrix, camera.distortion)
    return pixels.reshape(-1, 2)


def test_continuous_fit_does_not_reinitialize_each_good_frame(monkeypatch):
    estimator = HeadPoseEstimator(.063)
    camera = CalibrationProvider(70).at(1280, 720)
    first = estimator.solve(project(estimator, camera), camera, 1280, 1.)
    assert first.solver == 'reinitialize'

    def unexpected(*_, **__):
        pytest.fail('A valid continuous fit should not run RANSAC again')

    monkeypatch.setattr(cv2, 'solvePnPRansac', unexpected)
    for i in range(1, 20):
        pose = estimator.solve(project(estimator, camera, x=i * .002), camera, 1280, 1 + i / 30)
        assert pose.solver == 'continuous'
        assert pose.position == pytest.approx((i * .002, 0, .65), abs=1e-5)


def test_gap_and_camera_change_reinitialize_instead_of_reusing_stale_pose():
    estimator = HeadPoseEstimator(.063)
    provider = CalibrationProvider(70)
    camera = provider.at(1280, 720)
    estimator.solve(project(estimator, camera), camera, 1280, 1.)
    pose = estimator.solve(project(estimator, camera, .1), camera, 1280, 2.)
    assert pose.solver == 'reinitialize'
    other = provider.at(640, 360)
    pose = estimator.solve(project(estimator, other, .1), other, 640, 2.05)
    assert pose.solver == 'reinitialize'
    assert pose.position[0] == pytest.approx(.1, abs=1e-5)


def test_frozen_landmarks_produce_frozen_pose_with_model_mismatch():
    estimator = HeadPoseEstimator(.063)
    camera = CalibrationProvider(70).at(1280, 720)
    pixels = project(estimator, camera) + np.random.default_rng(9).normal(0, 1., (16, 2))
    positions = [estimator.solve(pixels, camera, 1280, i / 16).position for i in range(40)]
    assert np.max(np.ptp(positions[10:], axis=0)) < 1e-6


def test_bad_frame_does_not_poison_recovery_or_accept_reversed_time():
    estimator = HeadPoseEstimator(.063)
    camera = CalibrationProvider(70).at(1280, 720)
    pixels = project(estimator, camera)
    estimator.solve(pixels, camera, 1280, 1.)
    assert estimator.solve(pixels, camera, 1280, .9) is None
    assert estimator.solve(np.zeros((16, 2)), camera, 1280, 1.05) is None
    pose = estimator.solve(pixels, camera, 1280, 1.1)
    assert pose.solver == 'reinitialize'
    assert pose.position[2] == pytest.approx(.65, abs=1e-5)
