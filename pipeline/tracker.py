"""ByteTrack multi-object tracking with scene change detection.

Uses supervision's built-in ByteTrack implementation to maintain
player identity across frames. Resets tracker on scene changes
(camera cuts, replays) detected via frame difference threshold.
"""

from typing import Optional

import cv2
import numpy as np
import supervision as sv

from utils.logger import get_logger

logger = get_logger(__name__)


class PlayerTracker:
    """Tracks detected persons across frames using ByteTrack.

    Handles scene change detection to reset tracks when the camera
    cuts to a replay or different angle.

    Attributes:
        tracker: supervision ByteTrack instance.
        scene_change_threshold: Normalized frame difference to trigger reset.
        prev_frame_gray: Previous frame in grayscale for scene change detection.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        scene_change_threshold: float = 0.8,
    ):
        """Initialize the tracker.

        Args:
            max_age: Maximum number of frames a track can be lost before removal.
            min_hits: Minimum number of hits before a track is confirmed.
            scene_change_threshold: Normalized frame diff (0.0–1.0) to trigger reset.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=max_age,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=min_hits,
        )
        self.scene_change_threshold = scene_change_threshold
        self.prev_frame_gray: Optional[np.ndarray] = None

        logger.info(
            f"Tracker initialized: max_age={max_age}, "
            f"min_hits={min_hits}, "
            f"scene_threshold={scene_change_threshold}"
        )

    def _detect_scene_change(self, frame: np.ndarray) -> bool:
        """Detect if a scene change occurred between the current and previous frame.

        Uses mean absolute difference of grayscale frames, normalized to [0, 1].

        Args:
            frame: Current BGR frame.

        Returns:
            True if a scene change is detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return False

        # Compute normalized mean absolute difference
        diff = cv2.absdiff(self.prev_frame_gray, gray)
        score = float(np.mean(diff)) / 255.0

        self.prev_frame_gray = gray

        if score > self.scene_change_threshold:
            logger.info(f"Scene change detected (diff={score:.3f} > {self.scene_change_threshold})")
            return True

        return False

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> sv.Detections:
        """Update tracker with new detections.

        If a scene change is detected, the tracker is reset before processing
        the new detections.

        Args:
            detections: Current frame's detections from the detector.
            frame: Current BGR frame (used for scene change detection).

        Returns:
            sv.Detections with tracker_id assigned to each detection.
        """
        if self._detect_scene_change(frame):
            self.reset()

        if len(detections) == 0:
            return detections

        tracked = self.tracker.update_with_detections(detections)

        return tracked

    def reset(self) -> None:
        """Reset the tracker state (e.g., after a scene change)."""
        self.tracker.reset()
        logger.info("Tracker reset")
