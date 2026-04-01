"""Unit tests for the team classification module."""

import numpy as np
import pytest

from pipeline.team_classifier import TeamClassifier, TEAM_A, TEAM_B, TEAM_OTHER


class TestTeamClassifier:
    """Tests for TeamClassifier class."""

    def _create_colored_frame(self, colors, box_size=100):
        """Create a synthetic frame with colored rectangular regions.

        Args:
            colors: List of BGR tuples, one per "player".
            box_size: Size of each colored region in pixels.

        Returns:
            Tuple of (frame, bboxes array).
        """
        n = len(colors)
        frame_w = box_size * n
        frame_h = box_size * 2  # tall enough for torso extraction
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        bboxes = []
        for i, color in enumerate(colors):
            x1 = i * box_size
            x2 = x1 + box_size
            y1 = 0
            y2 = frame_h
            frame[y1:y2, x1:x2] = color
            bboxes.append([x1, y1, x2, y2])

        return frame, np.array(bboxes)

    def test_classify_two_teams(self):
        """Should separate two visually distinct jersey colors into different teams."""
        classifier = TeamClassifier(n_clusters=2, sample_region="full", bootstrap_frames=1)

        # 3 red players, 3 blue players
        colors = [
            (0, 0, 200),    # Red (BGR)
            (0, 0, 190),
            (0, 0, 210),
            (200, 0, 0),    # Blue (BGR)
            (190, 0, 0),
            (210, 0, 0),
        ]
        frame, bboxes = self._create_colored_frame(colors)
        labels = classifier.classify(frame, bboxes)

        assert len(labels) == 6
        # First 3 should be same team, last 3 should be same team
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_classify_three_clusters(self):
        """Should assign third cluster as 'other'."""
        classifier = TeamClassifier(n_clusters=3, sample_region="full", bootstrap_frames=1)

        colors = [
            (0, 0, 200), (0, 0, 200), (0, 0, 200),  # Red team
            (200, 0, 0), (200, 0, 0), (200, 0, 0),  # Blue team
            (0, 200, 200),                            # Yellow (referee)
        ]
        frame, bboxes = self._create_colored_frame(colors)
        labels = classifier.classify(frame, bboxes)

        assert len(labels) == 7
        # The single outlier should be classified as 'other'
        assert labels[6] == TEAM_OTHER

    def test_empty_bboxes(self):
        """Should return empty list for no detections."""
        classifier = TeamClassifier()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = classifier.classify(frame, np.array([]).reshape(0, 4))

        assert labels == []

    def test_too_few_detections(self):
        """Should label all as 'other' when fewer detections than clusters."""
        classifier = TeamClassifier(n_clusters=3)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 200)
        bboxes = np.array([[0, 0, 50, 100]])

        labels = classifier.classify(frame, bboxes)
        assert labels == [TEAM_OTHER]
