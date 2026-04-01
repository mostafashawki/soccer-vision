"""YOLOv8 person detection module.

Uses ultralytics YOLOv8n pretrained on COCO to detect persons (class 0).
Returns supervision Detections objects for downstream processing.
"""

from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger(__name__)

# COCO class ID for 'person'
PERSON_CLASS_ID = 0


class PlayerDetector:
    """Detects persons in video frames using YOLOv8.

    Attributes:
        model: Loaded YOLO model instance.
        confidence_threshold: Minimum confidence to keep a detection.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "auto",
        use_field_mask: bool = True,
    ):
        """Initialize the detector.

        Args:
            weights: Path to YOLO weights file.
            confidence_threshold: Minimum detection confidence (0.0–1.0).
            device: Inference device — 'auto', 'cpu', 'cuda', or 'cuda:0'.
        """
        self.confidence_threshold = confidence_threshold

        # Resolve device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading YOLO model: {weights} on {device}")
        self.model = YOLO(weights)
        self.device = device
        self.use_field_mask = use_field_mask
        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run person detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            sv.Detections containing only person-class detections
            above the confidence threshold.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if self.use_field_mask and len(detections) > 0:
            # Filter out detections not on green grass
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Define green range in HSV (OpenCV uses H: 0-179)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            h, w = frame.shape[:2]
            keep_mask = np.zeros(len(detections), dtype=bool)

            for i, xyxy in enumerate(detections.xyxy):
                x1, y1, x2, y2 = xyxy
                # Check the middle bottom of the bounding box (where the player's feet are)
                cx = int((x1 + x2) / 2)
                # Sample slightly above the bottom boundary to hit the pitch
                cy = int(y2 - (y2 - y1) * 0.1)

                cx = max(0, min(cx, w - 1))
                cy = max(0, min(cy, h - 1))

                if green_mask[cy, cx] > 0:
                    keep_mask[i] = True

            detections = detections[keep_mask]

        return detections
