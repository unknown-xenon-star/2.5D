from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class TrackingState:
    detected: bool
    x: float
    y: float
    z: float
    raw_x: float
    raw_y: float
    raw_z: float
    landmarks_px: Optional[List[Tuple[int, int]]]


class HeadTracker:
    """Webcam + MediaPipe face tracking and stabilized head-position estimates."""

    LEFT_EYE_IDX = 33
    RIGHT_EYE_IDX = 263

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        smoothing_alpha: float = 0.22,
        deadzone_xy: float = 0.015,
        deadzone_z: float = 0.01,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.01, 1.0))
        self.deadzone_xy = deadzone_xy
        self.deadzone_z = deadzone_z

        self._smoothed = np.zeros(3, dtype=np.float32)
        self._reference_eye_dist: Optional[float] = None

    def read(self) -> tuple[Optional[np.ndarray], TrackingState]:
        ok, frame = self.cap.read()
        if not ok:
            return None, TrackingState(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        raw = np.zeros(3, dtype=np.float32)
        detected = False
        landmarks_px: Optional[List[Tuple[int, int]]] = None

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lm = face.landmark

            left_eye = lm[self.LEFT_EYE_IDX]
            right_eye = lm[self.RIGHT_EYE_IDX]

            left_xy = np.array([left_eye.x * w, left_eye.y * h], dtype=np.float32)
            right_xy = np.array([right_eye.x * w, right_eye.y * h], dtype=np.float32)
            eye_center = 0.5 * (left_xy + right_xy)
            eye_dist = float(np.linalg.norm(right_xy - left_xy))

            if eye_dist > 1e-6:
                if self._reference_eye_dist is None:
                    self._reference_eye_dist = eye_dist
                else:
                    # Very slow adaptation keeps depth calibration stable over time.
                    self._reference_eye_dist = 0.99 * self._reference_eye_dist + 0.01 * eye_dist

                cx = w * 0.5
                cy = h * 0.5
                raw_x = (eye_center[0] - cx) / cx
                raw_y = (eye_center[1] - cy) / cy
                raw_z = eye_dist / (self._reference_eye_dist + 1e-6) - 1.0

                raw[0] = float(np.clip(raw_x, -1.2, 1.2))
                raw[1] = float(np.clip(raw_y, -1.2, 1.2))
                raw[2] = float(np.clip(raw_z, -1.0, 1.0))
                detected = True

            landmarks_px = [
                (int(left_xy[0]), int(left_xy[1])),
                (int(right_xy[0]), int(right_xy[1])),
                (int(eye_center[0]), int(eye_center[1])),
            ]

        if detected:
            self._smoothed = self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * self._smoothed
        else:
            # Relax toward center when detection is lost to avoid stale camera offsets.
            self._smoothed *= 0.95

        x = self._apply_deadzone(float(self._smoothed[0]), self.deadzone_xy)
        y = self._apply_deadzone(float(self._smoothed[1]), self.deadzone_xy)
        z = self._apply_deadzone(float(self._smoothed[2]), self.deadzone_z)

        state = TrackingState(
            detected=detected,
            x=x,
            y=y,
            z=z,
            raw_x=float(raw[0]),
            raw_y=float(raw[1]),
            raw_z=float(raw[2]),
            landmarks_px=landmarks_px,
        )
        return frame, state

    @staticmethod
    def _apply_deadzone(value: float, threshold: float) -> float:
        if abs(value) < threshold:
            return 0.0
        return value

    def draw_debug(self, frame: np.ndarray, state: TrackingState) -> np.ndarray:
        if state.landmarks_px:
            for pt in state.landmarks_px:
                cv2.circle(frame, pt, 4, (80, 255, 80), -1, cv2.LINE_AA)

        text_color = (230, 230, 230)
        cv2.putText(
            frame,
            f"raw xyz: {state.raw_x:+.3f} {state.raw_y:+.3f} {state.raw_z:+.3f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"smoothed xyz: {state.x:+.3f} {state.y:+.3f} {state.z:+.3f}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
        status = "face: detected" if state.detected else "face: not found"
        cv2.putText(
            frame,
            status,
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 120) if state.detected else (120, 120, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def close(self) -> None:
        self.cap.release()
        self.face_mesh.close()
