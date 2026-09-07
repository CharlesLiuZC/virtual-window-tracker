from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .config import Settings
from .calibration import CalibrationProvider
from .stabilization import StablePositionTracker
from .geometry import (
    Point2,
    average_points,
    estimate_viewer_position_m,
    pixel_distance,
    screen_normalized,
)

RIGHT_EYE_KEYPOINT = 0
LEFT_EYE_KEYPOINT = 1


def _point_payload(point: Point2, width: int, height: int) -> dict[str, Any]:
    normalized = screen_normalized(point, width, height)
    return {
        "pixel": {"x": round(point.x, 3), "y": round(point.y, 3)},
        "screen_normalized": {
            "x": round(normalized.x, 6),
            "y": round(normalized.y, 6),
        },
    }


class FacePositionTracker:
    def __init__(self, settings: Settings, model_path: Path) -> None:
        self.settings = settings
        self._camera = CalibrationProvider(settings.camera_hfov_deg, settings.calibration_path)
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=settings.min_detection_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self._stable = StablePositionTracker(
            settings.filter_min_cutoff,
            settings.filter_beta,
            settings.filter_derivative_cutoff,
        )
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FacePositionTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict[str, Any]:
        height, width = frame_bgr.shape[:2]
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(image, timestamp_ms)

        camera = self._camera.at(width, height)
        intrinsics = camera.intrinsics
        observations = []
        for candidate in sorted(result.detections or [], key=lambda d:
                                d.bounding_box.width * d.bounding_box.height, reverse=True):
            points = candidate.keypoints or []
            if len(points) < 2 or any(p.x is None or p.y is None for p in points[:2]):
                continue
            right = Point2(points[RIGHT_EYE_KEYPOINT].x * width, points[RIGHT_EYE_KEYPOINT].y * height)
            left = Point2(points[LEFT_EYE_KEYPOINT].x * width, points[LEFT_EYE_KEYPOINT].y * height)
            center = average_points([left, right])
            distance = pixel_distance(left, right)
            if distance < 8:  # Tiny detections make monocular depth ill-conditioned.
                continue
            corrected = camera.undistort(np.array([[right.x, right.y], [left.x, left.y]]))
            corrected_center = Point2(*corrected.mean(axis=0))
            position = estimate_viewer_position_m(corrected_center, float(np.linalg.norm(corrected[0] - corrected[1])),
                                                  intrinsics, self.settings.assumed_ipd_m)
            if position is not None:
                observations.append((candidate, left, right, center, distance, position))
        accepted = self._stable.update([item[5] for item in observations], timestamp_ms / 1000.0)
        if accepted is None:
            return {"tracking": False, "track_id": self._stable.track_id, "face": None}
        index, filtered_position = accepted
        detection, left_eye, right_eye, eye_center, eye_distance, raw_position = observations[index]

        bbox = detection.bounding_box
        min_x = float(bbox.origin_x)
        min_y = float(bbox.origin_y)
        box_width = float(bbox.width)
        box_height = float(bbox.height)

        return {
            "tracking": True,
            "track_id": self._stable.track_id,
            "face": {
                "bbox": {
                    "pixel": {
                        "x": round(min_x, 3),
                        "y": round(min_y, 3),
                        "width": round(box_width, 3),
                        "height": round(box_height, 3),
                    },
                    "normalized": {
                        "x": round(min_x / width, 6),
                        "y": round(min_y / height, 6),
                        "width": round(box_width / width, 6),
                        "height": round(box_height / height, 6),
                    },
                },
                "eyes": {
                    "left": _point_payload(left_eye, width, height),
                    "right": _point_payload(right_eye, width, height),
                    "center": _point_payload(eye_center, width, height),
                    "distance_pixels": round(eye_distance, 3),
                },
                "viewer_position_m": (
                    {
                        "raw": {
                            "x": round(raw_position[0], 6),
                            "y": round(raw_position[1], 6),
                            "z": round(raw_position[2], 6),
                        },
                        "filtered": {
                            "x": round(filtered_position[0], 6),
                            "y": round(filtered_position[1], 6),
                            "z": round(filtered_position[2], 6),
                        },
                        "coordinate_system": "x-right_y-up_z-toward-viewer",
                        "calibrated": False,
                        "intrinsics_calibrated": intrinsics.calibrated,
                        "method": "eye-distance-assumed-ipd",
                    }
                    if raw_position is not None and filtered_position is not None
                    else None
                ),
                "head_rotation_deg": None,
                "facial_transformation_matrix": None,
            },
        }


def draw_debug_overlay(frame: np.ndarray, result: dict[str, Any], fps: float) -> None:
    face = result.get("face")
    if result.get("tracking") and face:
        bbox = face["bbox"]["pixel"]
        start = (int(bbox["x"]), int(bbox["y"]))
        end = (int(bbox["x"] + bbox["width"]), int(bbox["y"] + bbox["height"]))
        cv2.rectangle(frame, start, end, (80, 220, 80), 2)
        for key, color in (("left", (255, 160, 40)), ("right", (40, 160, 255))):
            point = face["eyes"][key]["pixel"]
            cv2.circle(frame, (int(point["x"]), int(point["y"])), 5, color, -1)
        center = face["eyes"]["center"]["pixel"]
        cv2.drawMarker(
            frame,
            (int(center["x"]), int(center["y"])),
            (50, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        position = face.get("viewer_position_m")
        if position:
            p = position["filtered"]
            label = f"viewer x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f} m"
            cv2.putText(frame, label, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 255), 2)
    else:
        cv2.putText(frame, "face lost", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 255), 2)
    cv2.putText(frame, f"FPS {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 80), 2)
