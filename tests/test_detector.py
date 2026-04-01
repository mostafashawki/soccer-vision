"""Unit tests for the player detection module."""

import numpy as np
import pytest


class TestPlayerDetector:
    """Tests for PlayerDetector class."""

    def test_detect_returns_detections(self):
        """Detector should return sv.Detections for a valid frame."""
        from pipeline.detector import PlayerDetector
        import supervision as sv

        detector = PlayerDetector(
            weights="yolov8n.pt",
            confidence_threshold=0.3,
            device="cpu",
        )

        # Create a synthetic frame (black 640x480 image)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert isinstance(detections, sv.Detections)

    def test_detect_empty_frame_no_crash(self):
        """Detector should not crash on a blank frame."""
        from pipeline.detector import PlayerDetector

        detector = PlayerDetector(
            weights="yolov8n.pt",
            confidence_threshold=0.5,
            device="cpu",
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        # No persons in a black frame
        assert len(detections) == 0

    def test_confidence_threshold_filters(self):
        """Higher confidence threshold should return fewer or equal detections."""
        from pipeline.detector import PlayerDetector

        detector_low = PlayerDetector(
            weights="yolov8n.pt",
            confidence_threshold=0.1,
            device="cpu",
        )
        detector_high = PlayerDetector(
            weights="yolov8n.pt",
            confidence_threshold=0.9,
            device="cpu",
        )

        # Create a frame with some visual content
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections_low = detector_low.detect(frame)
        detections_high = detector_high.detect(frame)

        assert len(detections_high) <= len(detections_low)
