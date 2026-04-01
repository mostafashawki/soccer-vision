"""Per-frame count aggregation and confidence scoring.

Combines detection, tracking, and team classification results
into structured FrameResult objects and generates the final GameReport.
"""

from typing import List, Optional

import numpy as np
import supervision as sv

from utils.schema import FrameResult, GameReport, Summary
from utils.logger import get_logger

logger = get_logger(__name__)


class Aggregator:
    """Aggregates per-frame detection and classification results.

    Produces FrameResult objects containing player counts per team
    and a confidence score derived from detection confidences.
    """

    def aggregate_frame(
        self,
        frame_id: int,
        fps: float,
        detections: sv.Detections,
        team_labels: List[str],
    ) -> FrameResult:
        """Aggregate a single frame's results into a FrameResult.

        Args:
            frame_id: Frame index in the source video.
            fps: Source video frames per second.
            detections: Tracked detections for this frame.
            team_labels: Team label per detection ('team_a', 'team_b', 'other').

        Returns:
            FrameResult with counts and confidence for this frame.
        """
        team_a_count = sum(1 for label in team_labels if label == "team_a")
        team_b_count = sum(1 for label in team_labels if label == "team_b")
        other_count = sum(1 for label in team_labels if label == "other")

        # Confidence = average detection confidence across all persons in frame
        if len(detections) > 0 and detections.confidence is not None:
            confidence = float(np.mean(detections.confidence))
        else:
            confidence = 0.0

        timestamp_sec = round(frame_id / fps, 2) if fps > 0 else 0.0

        return FrameResult(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            team_a_count=team_a_count,
            team_b_count=team_b_count,
            other_count=other_count,
            confidence=round(confidence, 3),
        )

    def build_report(
        self,
        game_id: str,
        total_frames: int,
        fps: float,
        frame_results: List[FrameResult],
    ) -> GameReport:
        """Build the complete game report from all frame results.

        Args:
            game_id: Identifier for this game/video.
            total_frames: Total frame count in the source video.
            fps: Source video FPS.
            frame_results: List of all processed FrameResult objects.

        Returns:
            Complete GameReport with summary statistics computed.
        """
        report = GameReport(
            game_id=game_id,
            total_frames=total_frames,
            fps=fps,
            frames=frame_results,
        )
        report.compute_summary()

        logger.info(
            f"Report built: {len(frame_results)} frames processed, "
            f"avg_team_a={report.summary.avg_team_a}, "
            f"avg_team_b={report.summary.avg_team_b}, "
            f"avg_confidence={report.summary.avg_confidence}"
        )

        return report
