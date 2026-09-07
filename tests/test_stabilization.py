import math
import statistics

import pytest

from face_tracker.filtering import OneEuroFilter
from face_tracker.geometry import CameraIntrinsics, Point2, estimate_viewer_position_m
from face_tracker.stabilization import StablePositionTracker


def test_face_order_changes_do_not_switch_target():
    tracker = StablePositionTracker()
    a, b = (0.0, 0.0, 0.6), (0.5, 0.0, 0.9)
    assert tracker.update([a, b], 0)[0] == 0
    assert tracker.update([b, a], 1 / 30)[0] == 1
    assert tracker.track_id == 1


def test_short_dropout_keeps_filter_and_identity():
    tracker = StablePositionTracker()
    tracker.update([(0, 0, 0.6)], 0)
    assert tracker.update([], 0.033) is None
    _, filtered = tracker.update([(0.01, 0, 0.6)], 0.1)
    assert 0 < filtered[0] < 0.01
    assert tracker.track_id == 1


def test_long_dropout_reacquires_with_new_identity():
    tracker = StablePositionTracker()
    tracker.update([(0, 0, 0.6)], 0)
    assert tracker.update([(0.4, 0, 0.7)], 1)[1] == (0.4, 0, 0.7)
    assert tracker.track_id == 2


def test_single_frame_spike_and_reordered_time_are_rejected():
    tracker = StablePositionTracker()
    tracker.update([(0, 0, 0.6)], 1)
    assert tracker.update([(0, 0, 1.4)], 1.033) is None
    assert tracker.update([(0, 0, 0.6)], 0.9) is None
    assert tracker.update([(0.002, 0, 0.6)], 1.066) is not None
    assert tracker.track_id == 1


@pytest.mark.parametrize('z', [math.nan, math.inf, 0, -1, 5])
def test_invalid_metric_positions_rejected(z):
    assert StablePositionTracker().update([(0, 0, z)], 0) is None


def test_invalid_geometry_rejected():
    camera = CameraIntrinsics.from_horizontal_fov(1280, 720, 70)
    assert estimate_viewer_position_m(Point2(math.nan, 0), 40, camera, 0.063) is None
    with pytest.raises(ValueError):
        CameraIntrinsics.from_horizontal_fov(1280, 720, 0)


def test_adaptive_filter_motion_lag_and_stationary_jitter():
    old, new = OneEuroFilter(beta=0.035), OneEuroFilter(beta=4)
    target = [0.0] * 30 + [(i + 1) * 0.006 for i in range(30)]
    old_result = [old.apply(p, i / 30) for i, p in enumerate(target)]
    new_result = [new.apply(p, i / 30) for i, p in enumerate(target)]
    old_error = statistics.mean(abs(target[i] - old_result[i]) for i in range(40, 60))
    new_error = statistics.mean(abs(target[i] - new_result[i]) for i in range(40, 60))
    assert new_error < old_error * 0.8
    noise = [0.6 + (0.003 if i % 2 else -0.003) for i in range(180)]
    tracker = StablePositionTracker()
    filtered = [tracker.update([(0, 0, p)], i / 30)[1][2] for i, p in enumerate(noise)]
    assert statistics.pstdev(filtered[60:]) < statistics.pstdev(noise[60:]) * 0.25
