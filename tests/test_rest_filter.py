import math

import numpy as np
import pytest

from face_tracker.filtering import RestAwarePositionFilter
from face_tracker.stabilization import StablePositionTracker


def rested():
    f = RestAwarePositionFilter(1.2, 4, 1)
    for i in range(32):
        f.apply(0., 0., .65, i / 16)
    assert f.stationary
    return f


@pytest.mark.parametrize('hz', [8, 16, 30, 60])
def test_stationary_noise_does_not_move_output(hz):
    f = RestAwarePositionFilter(1.2, 4, 1)
    positions = [f.apply(.0006 * math.sin(i), .0005 * math.cos(i),
                        .65 + .0015 * math.sin(2 * i), i / hz) for i in range(hz * 4)]
    assert f.stationary
    assert np.max(np.ptp(positions[hz * 2:], axis=0)) == 0


def test_single_frame_two_centimeter_spike_cannot_release_static_anchor():
    f = rested()
    before = f.output
    assert f.apply(.02, 0., .65, 2.) == before
    assert f.apply(0., 0., .65, 2.0625) == before
    assert f.stationary


def test_confirmed_motion_releases_without_snapping_to_measurement():
    f = rested()
    before = f.output
    assert f.apply(.04, 0., .65, 2.) == before
    result = f.apply(.04, 0., .65, 2.0625)
    assert 0 < result[0] < .04
    assert not f.stationary
    for i in range(2, 20):
        result = f.apply(.04, 0., .65, 2 + i / 16)
    assert result[0] == pytest.approx(.04, abs=.001)


def test_slow_motion_cannot_be_absorbed_by_a_drifting_rest_reference():
    f = rested()
    for i in range(1, 49):
        result = f.apply(i * .0005, 0., .65, 2 + i / 16)
    assert result[0] > .02


def test_opposite_direction_spikes_do_not_count_as_confirmed_motion():
    f = rested()
    before = f.output
    assert f.apply(.02, 0., .65, 2.) == before
    assert f.apply(-.02, 0., .65, 2.0625) == before
    assert f.apply(0., 0., .65, 2.125) == before


def test_gap_and_identity_reset_clear_static_anchor():
    f = rested()
    assert f.apply(.04, 0., .65, 3.)[0] > 0
    assert not f.stationary
    tracker = StablePositionTracker()
    for i in range(32):
        tracker.update([(0., 0., .65)], i / 16)
    assert tracker.stationary
    assert tracker.update([(.2, 0., .8)], 4.)[1] == (.2, 0., .8)
    assert tracker.track_id == 2
    assert not tracker.stationary
