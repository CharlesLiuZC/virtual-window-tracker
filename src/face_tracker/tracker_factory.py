from .config import LANDMARK_MODEL_URL, Settings
from .model_loader import ensure_model
from .tracker import FacePositionTracker


def create_tracker(settings: Settings):
    if settings.tracker_backend == 'landmarker':
        from .landmark_tracker import LandmarkPositionTracker
        path = ensure_model(settings.landmark_model_path, LANDMARK_MODEL_URL)
        return LandmarkPositionTracker(settings, path)
    if settings.tracker_backend != 'detector':
        raise ValueError('FACE_TRACKER_BACKEND must be detector or landmarker')
    path = ensure_model(settings.model_path, settings.model_url)
    return FacePositionTracker(settings, path)
