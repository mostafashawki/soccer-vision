"""Annotated video renderer using supervision annotators.

Draws bounding boxes with team-colored labels and a live player
count overlay on each video frame.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv

from utils.logger import get_logger

logger = get_logger(__name__)

# Team colors (BGR format for OpenCV)
TEAM_COLORS: Dict[str, Tuple[int, int, int]] = {
    "team_a": (0, 120, 255),    # Orange
    "team_b": (255, 100, 0),    # Blue
    "other": (128, 128, 128),   # Gray
}

# Supervision Color palette
TEAM_SV_COLORS: Dict[str, sv.Color] = {
    "team_a": sv.Color(r=255, g=120, b=0),
    "team_b": sv.Color(r=0, g=100, b=255),
    "other": sv.Color(r=128, g=128, b=128),
}


class Renderer:
    """Renders annotated video frames with bounding boxes and team overlay.

    Uses supervision BoxAnnotator and LabelAnnotator for clean, professional
    annotations, plus a custom counter overlay showing live team counts.
    """

    def __init__(self):
        """Initialize annotation tools."""
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=5,
        )
        logger.info("Renderer initialized")

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        team_labels: List[str],
        frame_id: int,
        team_a_count: int,
        team_b_count: int,
        other_count: int,
        confidence: float,
    ) -> np.ndarray:
        """Annotate a single frame with bounding boxes and overlay.

        Args:
            frame: BGR frame to annotate (will be copied, not modified in-place).
            detections: Tracked detections for this frame.
            team_labels: Team label per detection.
            frame_id: Current frame index.
            team_a_count: Number of Team A players.
            team_b_count: Number of Team B players.
            other_count: Number of other persons.
            confidence: Average detection confidence for this frame.

        Returns:
            Annotated BGR frame as numpy array.
        """
        annotated = frame.copy()

        if len(detections) > 0 and len(team_labels) > 0:
            # Assign colors per detection based on team
            colors = [TEAM_SV_COLORS.get(label, TEAM_SV_COLORS["other"]) for label in team_labels]

            # Create labels for each detection
            labels = []
            for i, label in enumerate(team_labels):
                conf = detections.confidence[i] if detections.confidence is not None else 0.0
                display = label.replace("_", " ").title()
                labels.append(f"{display} {conf:.0%}")

            # Annotate boxes with per-detection colors
            # Use class_id to drive colors via ColorPalette
            class_ids = np.array([
                0 if l == "team_a" else 1 if l == "team_b" else 2
                for l in team_labels
            ])
            detections.class_id = class_ids

            color_palette = sv.ColorPalette(colors=[
                TEAM_SV_COLORS["team_a"],
                TEAM_SV_COLORS["team_b"],
                TEAM_SV_COLORS["other"],
            ])

            box_annotator = sv.BoxAnnotator(
                thickness=2,
                color=color_palette,
            )
            label_annotator = sv.LabelAnnotator(
                text_scale=0.5,
                text_thickness=1,
                text_padding=5,
                color=color_palette,
            )

            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

        # Draw counter overlay
        annotated = self._draw_counter_overlay(
            annotated, frame_id, team_a_count, team_b_count, other_count, confidence
        )

        return annotated

    def _draw_counter_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
        team_a_count: int,
        team_b_count: int,
        other_count: int,
        confidence: float,
    ) -> np.ndarray:
        """Draw a semi-transparent counter overlay in the top-left corner.

        Args:
            frame: BGR frame to draw on.
            frame_id: Current frame index.
            team_a_count: Number of Team A players.
            team_b_count: Number of Team B players.
            other_count: Number of other persons.
            confidence: Average detection confidence.

        Returns:
            Frame with the overlay drawn.
        """
        overlay = frame.copy()

        # Overlay background
        padding = 15
        line_height = 28
        num_lines = 5
        box_w = 280
        box_h = padding * 2 + line_height * num_lines

        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Text lines
        x = 10 + padding
        y = 10 + padding + 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        lines = [
            (f"Frame: {frame_id}", (255, 255, 255)),
            (f"Team A: {team_a_count}", TEAM_COLORS["team_a"]),
            (f"Team B: {team_b_count}", TEAM_COLORS["team_b"]),
            (f"Other:  {other_count}", TEAM_COLORS["other"]),
            (f"Confidence: {confidence:.0%}", (255, 255, 255)),
        ]

        for text, color in lines:
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height

        return frame
