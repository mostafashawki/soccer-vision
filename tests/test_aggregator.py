"""Unit tests for the aggregation module."""

import numpy as np
import pytest
import supervision as sv

from pipeline.aggregator import Aggregator
from utils.schema import FrameResult, GameReport


class TestAggregator:
    """Tests for the Aggregator class."""

    def _make_detections(self, n: int, confidence: float = 0.9) -> sv.Detections:
        """Create mock detections with N bounding boxes."""
        if n == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array([[i * 100, 0, i * 100 + 80, 200] for i in range(n)]),
            confidence=np.full(n, confidence),
        )

    def test_aggregate_frame_counts(self):
        """Should correctly count players per team."""
        agg = Aggregator()
        detections = self._make_detections(5, confidence=0.85)
        labels = ["team_a", "team_a", "team_b", "team_b", "other"]

        result = agg.aggregate_frame(
            frame_id=10,
            fps=30.0,
            detections=detections,
            team_labels=labels,
        )

        assert isinstance(result, FrameResult)
        assert result.frame_id == 10
        assert result.team_a_count == 2
        assert result.team_b_count == 2
        assert result.other_count == 1
        assert result.confidence == 0.85
        assert result.timestamp_sec == pytest.approx(10 / 30.0, rel=0.01)

    def test_aggregate_frame_empty(self):
        """Should handle empty detections gracefully."""
        agg = Aggregator()
        detections = self._make_detections(0)

        result = agg.aggregate_frame(
            frame_id=0,
            fps=30.0,
            detections=detections,
            team_labels=[],
        )

        assert result.team_a_count == 0
        assert result.team_b_count == 0
        assert result.other_count == 0
        assert result.confidence == 0.0

    def test_build_report(self):
        """Should compute correct summary statistics."""
        agg = Aggregator()

        frame_results = [
            FrameResult(frame_id=0, timestamp_sec=0.0, team_a_count=4, team_b_count=3, other_count=1, confidence=0.9),
            FrameResult(frame_id=1, timestamp_sec=0.033, team_a_count=5, team_b_count=4, other_count=0, confidence=0.8),
            FrameResult(frame_id=2, timestamp_sec=0.067, team_a_count=3, team_b_count=3, other_count=2, confidence=0.85),
        ]

        report = agg.build_report(
            game_id="test_game",
            total_frames=100,
            fps=30.0,
            frame_results=frame_results,
        )

        assert isinstance(report, GameReport)
        assert report.game_id == "test_game"
        assert report.total_frames == 100
        assert len(report.frames) == 3
        assert report.summary.avg_team_a == pytest.approx(4.0, rel=0.01)
        assert report.summary.avg_team_b == pytest.approx(3.33, rel=0.01)
        assert report.summary.avg_confidence == pytest.approx(0.85, rel=0.01)

    def test_build_report_empty(self):
        """Should handle empty frame results."""
        agg = Aggregator()
        report = agg.build_report("empty", 0, 30.0, [])

        assert len(report.frames) == 0
        assert report.summary.avg_team_a == 0.0
        assert report.summary.avg_team_b == 0.0
