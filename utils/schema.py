"""Pydantic models for pipeline output schema and configuration."""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class FrameResult(BaseModel):
    """Per-frame analytics result."""

    frame_id: int = Field(..., description="Frame index in the source video")
    timestamp_sec: float = Field(..., description="Timestamp in seconds")
    team_a_count: int = Field(0, ge=0, description="Number of Team A players detected")
    team_b_count: int = Field(0, ge=0, description="Number of Team B players detected")
    other_count: int = Field(0, ge=0, description="Number of other persons (referee, goalkeeper, etc.)")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Average detection confidence for this frame")


class Summary(BaseModel):
    """Aggregate summary statistics across all processed frames."""

    avg_team_a: float = Field(0.0, description="Average Team A count per frame")
    avg_team_b: float = Field(0.0, description="Average Team B count per frame")
    avg_confidence: float = Field(0.0, description="Average confidence across all frames")
    team_a_color: str = Field("", description="Dominant jersey color for Team A")
    team_b_color: str = Field("", description="Dominant jersey color for Team B")


class GameReport(BaseModel):
    """Complete game analytics report."""

    game_id: str = Field(..., description="Identifier for the game/video")
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of when processing completed",
    )
    total_frames: int = Field(0, ge=0, description="Total frames in source video")
    fps: float = Field(0.0, description="Source video frames per second")
    frames: List[FrameResult] = Field(default_factory=list, description="Per-frame results")
    summary: Summary = Field(default_factory=Summary, description="Aggregate statistics")

    def compute_summary(self) -> None:
        """Compute summary statistics from frame results."""
        if not self.frames:
            return

        n = len(self.frames)
        self.summary = Summary(
            avg_team_a=round(sum(f.team_a_count for f in self.frames) / n, 2),
            avg_team_b=round(sum(f.team_b_count for f in self.frames) / n, 2),
            avg_confidence=round(sum(f.confidence for f in self.frames) / n, 2),
        )

    def to_summary_text(self) -> str:
        """Generate human-readable summary text for non-technical stakeholders."""
        self.compute_summary()
        team_a_label = f"Team A ({self.summary.team_a_color})" if self.summary.team_a_color else "Team A"
        team_b_label = f"Team B ({self.summary.team_b_color})" if self.summary.team_b_color else "Team B"

        lines = [
            f"Soccer Vision — Game Report",
            f"{'=' * 40}",
            f"Game ID:        {self.game_id}",
            f"Processed at:   {self.processed_at}",
            f"Total frames:   {self.total_frames}",
            f"Source FPS:      {self.fps}",
            f"Frames analyzed: {len(self.frames)}",
            f"",
            f"Detected Jersey Colors:",
            f"  {team_a_label}",
            f"  {team_b_label}",
            f"",
            f"Average Players Visible:",
            f"  Team A:       {self.summary.avg_team_a:.1f}",
            f"  Team B:       {self.summary.avg_team_b:.1f}",
            f"  Avg Confidence: {self.summary.avg_confidence:.1%}",
        ]
        return "\n".join(lines)


class AppConfig(BaseModel):
    """Application configuration loaded from config.yaml."""

    class ModelConfig(BaseModel):
        weights: str = "yolov8n.pt"
        confidence_threshold: float = 0.5
        field_mask: bool = True
        device: str = "auto"

    class TrackingConfig(BaseModel):
        max_age: int = 30
        min_hits: int = 3
        scene_change_threshold: float = 0.8

    class TeamClassificationConfig(BaseModel):
        n_clusters: int = 4
        color_space: str = "HSV"
        sample_region: str = "torso"
        bootstrap_frames: int = 50

    class OutputConfig(BaseModel):
        output_dir: str = "./output"
        save_video: bool = True
        save_json: bool = True
        save_predictions: bool = False
        frame_skip: int = 1
        blur_threshold: int = 0  # Laplacian variance below this value → frame skipped. 0 = disabled.
        log_level: str = "INFO"

    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    team_classification: TeamClassificationConfig = Field(default_factory=TeamClassificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            AppConfig instance with values from the file, defaults for missing keys.
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)
