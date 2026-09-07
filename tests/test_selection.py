from face_tracker.selection import FaceSelector


def test_keeps_viewer_when_detection_order_changes():
    selector = FaceSelector()
    viewer = (0.1, 0.1, 0.3, 0.3)
    other = (0.7, 0.1, 0.2, 0.2)
    assert selector.select([other, viewer], 0) == 1
    assert selector.select([viewer, other], 0.1) == 0


def test_waits_before_switching_to_distant_face():
    selector = FaceSelector()
    assert selector.select([(0.1, 0.1, 0.2, 0.2)], 0) == 0
    other = [(0.7, 0.1, 0.2, 0.2)]
    assert selector.select(other, 0.2) is None
    assert selector.select([], 0.3) is None
    assert selector.select(other, 0.8) == 0
