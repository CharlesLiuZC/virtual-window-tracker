import json
import math
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_tracker.calibration import CalibrationProvider
from face_tracker.capture import CapturedFrame, LatestCamera, LatestFrameSlot
from face_tracker.geometry import Point2, estimate_viewer_position_m
from face_tracker.head_pose import HeadPoseEstimator
from face_tracker.service import TrackingService
from face_tracker.config import Settings


@pytest.mark.parametrize('yaw', [-45, -25, 0, 25, 45])
def test_rotation_does_not_change_synthetic_eye_depth(yaw):
    estimator = HeadPoseEstimator(0.063)
    camera = CalibrationProvider(70).at(1280, 720)
    points, _ = cv2.projectPoints(estimator.object_points, np.array([0., math.radians(yaw), 0.]),
                                 np.array([0.03, -0.02, 0.65]), camera.matrix, camera.distortion)
    pose = estimator.solve(points.reshape(-1, 2), camera, 1280)
    assert pose is not None
    assert pose.position == pytest.approx((0.03, 0.02, 0.65), abs=1e-5)
    assert pose.angles['yaw'] == pytest.approx(yaw, abs=0.01)


def test_pose_rejects_large_turn_and_invalid_points():
    estimator = HeadPoseEstimator(0.063)
    camera = CalibrationProvider(70).at(1280, 720)
    assert estimator.solve(np.full((16, 2), np.nan), camera, 1280) is None
    points, _ = cv2.projectPoints(estimator.object_points, np.array([0., math.radians(75), 0.]),
                                 np.array([0., 0., 0.65]), camera.matrix, camera.distortion)
    assert estimator.solve(points.reshape(-1, 2), camera, 1280) is None


def test_pnp_rejects_outliers_and_recovers_noisy_pose():
    estimator = HeadPoseEstimator(0.063)
    camera = CalibrationProvider(70).at(1280, 720)
    pixels, _ = cv2.projectPoints(estimator.object_points, np.array([.1, .3, 0.]),
                                 np.array([0., 0., .65]), camera.matrix, camera.distortion)
    pixels = pixels.reshape(-1, 2) + np.random.default_rng(8).normal(0, .4, (16, 2))
    pixels[[0, 10]] += 70
    pose = estimator.solve(pixels, camera, 1280)
    assert pose is not None
    assert pose.position[2] == pytest.approx(.65, abs=.008)
    assert pose.inlier_ratio < 1


def test_eye_distance_depth_is_biased_when_yawing():
    estimator = HeadPoseEstimator(.063)
    camera = CalibrationProvider(70).at(1280, 720)
    p, _ = cv2.projectPoints(estimator.object_points, np.array([0., math.radians(45), 0.]),
                            np.array([0., 0., .65]), camera.matrix, camera.distortion)
    p = p.reshape(-1, 2)
    a, b = p[[6, 7]].mean(axis=0), p[[8, 9]].mean(axis=0)
    old = estimate_viewer_position_m(Point2(*((a + b) / 2)), np.linalg.norm(a - b), camera.intrinsics, .063)
    assert old[2] - .65 > .15
    assert estimator.solve(p, camera, 1280).position[2] == pytest.approx(.65, abs=1e-5)


def test_calibration_scales_intrinsics_and_rejects_crop(tmp_path):
    path = tmp_path / 'camera.json'
    path.write_text(json.dumps(dict(width=1280, height=720, fx=900, fy=901, cx=640, cy=360,
                                    distortion=[.1, -.05, 0, 0, 0])))
    provider = CalibrationProvider(70, path)
    camera = provider.at(640, 360)
    assert camera.intrinsics.fx == 450
    assert camera.intrinsics.calibrated
    assert np.isfinite(camera.undistort(np.array([[200, 200]]))).all()
    with pytest.raises(ValueError, match='aspect ratio'):
        provider.at(640, 480)


def test_latest_slot_drops_old_frames_not_new_frames():
    slot = LatestFrameSlot()
    slot.put('old'); slot.put('middle'); slot.put('latest')
    assert slot.take(0) == 'latest'
    assert slot.take(0) is None
    assert slot.dropped == 2
    slot.put('stale'); slot.clear()
    assert slot.take(0) is None


def test_capture_epoch_prevents_using_a_previous_connection():
    camera = LatestCamera(lambda: None, threading.Event(), lambda *a, **k: None)
    camera.connected, camera.epoch = True, 2
    sample = CapturedFrame(None, 0, 0, 1, 30)
    assert not camera.is_current(sample)
    sample.epoch = 2
    assert camera.is_current(sample)
    camera.connected = False
    assert not camera.is_current(sample)


def test_service_does_not_publish_stale_cache():
    service = TrackingService(Settings())
    service._latest = {'tracking': True}
    service._published_monotonic = time.monotonic() - 1
    assert service.latest() is None


def test_example_calibration_is_not_reported_as_measured(tmp_path):
    path = tmp_path / 'example.json'
    path.write_text(json.dumps({'example_only': True}))
    with pytest.raises(ValueError, match='not measured'):
        CalibrationProvider(70, path).at(1280, 720)


def test_disconnection_clears_cache_immediately():
    service = TrackingService(Settings())
    service._latest = {'tracking': True}
    service._published_monotonic = time.monotonic()
    service._set_status('reconnecting')
    assert service.latest() is None


def test_landmarker_preserves_v1_geometry_payload_without_loading_model(monkeypatch):
    from face_tracker import landmark_tracker as module
    from face_tracker.stabilization import StablePositionTracker

    tracker = object.__new__(module.LandmarkPositionTracker)
    tracker.camera = CalibrationProvider(70)
    tracker.pose = HeadPoseEstimator(.063)
    tracker.stable = StablePositionTracker()
    tracker.last_timestamp = -1
    camera = tracker.camera.at(1280, 720)
    points, _ = cv2.projectPoints(tracker.pose.object_points, np.zeros(3),
                                 np.array([0., 0., .65]), camera.matrix, camera.distortion)
    pixels = np.tile([640., 360.], (478, 1))
    pixels[module.LANDMARK_IDS] = points.reshape(-1, 2)
    result = SimpleNamespace(face_landmarks=[[
        SimpleNamespace(x=x / 1280, y=y / 720) for x, y in pixels]])
    tracker.detector = SimpleNamespace(detect_for_video=lambda *_: result)
    monkeypatch.setattr(module.mp, 'Image', lambda **kwargs: kwargs)
    packet = tracker.process(np.zeros((720, 1280, 3), dtype=np.uint8), 1000)
    assert packet['tracking'] and packet['calibration_ready']
    face = packet['face']
    assert face['bbox']['normalized']['width'] == pytest.approx(face['bbox']['pixel']['width'] / 1280)
    assert face['facial_transformation_matrix'] is None
    assert face['viewer_position_m']['calibrated'] is False
    assert face['viewer_position_m']['filtered']['z'] == pytest.approx(.65, abs=1e-5)
    result.face_landmarks = []
    assert tracker.process(np.zeros((720, 1280, 3), dtype=np.uint8), 1100)['quality_reason'] == 'face_not_found'


def test_capture_thread_releases_device_and_keeps_only_latest_frame():
    waiting = threading.Event()
    stop = threading.Event()
    released = threading.Event()
    states = []

    class FakeCapture:
        count = 0

        def get(self, _):
            return 10

        def read(self):
            self.count += 1
            if self.count <= 3:
                return True, self.count
            waiting.set()
            stop.wait(2)
            return False, None

        def release(self):
            released.set()

    with LatestCamera(FakeCapture, stop, lambda state, **_: states.append(state)) as camera:
        assert waiting.wait(2)
        sample = camera.slot.take(0)
        assert sample.image == 3
        assert camera.slot.dropped == 2
        assert camera.is_current(sample)
        assert sample.monotonic_s <= time.monotonic()
    assert released.wait(1)
    assert not camera.thread.is_alive()
    assert not camera.is_current(sample)
    assert states == ['running']


def test_video_file_replay_is_paced_not_drained_at_decode_speed():
    stop, ready = threading.Event(), threading.Event()
    read_times = []

    class FakeFile:
        def get(self, _):
            return 20  # 50 ms between replay frames

        def read(self):
            read_times.append(time.monotonic())
            if len(read_times) == 3:
                ready.set()
            return True, len(read_times)

        def release(self):
            pass

    with LatestCamera(FakeFile, stop, lambda *_, **__: None, pace_file=True):
        assert ready.wait(2)
    assert read_times[2] - read_times[0] >= .08
