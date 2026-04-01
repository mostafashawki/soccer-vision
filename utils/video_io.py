"""Video I/O utilities — reading, writing, validation, and preprocessing."""

import os
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
MAX_RESOLUTION = 1080  # auto-downscale width to this if larger


def validate_video(path: str) -> dict:
    """Validate a video file and return its metadata.

    Args:
        path: Path to the video file.

    Returns:
        Dictionary with keys: width, height, fps, total_frames, duration_sec.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or cannot be opened.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}. File may be corrupted.")

    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    metadata["duration_sec"] = round(
        metadata["total_frames"] / metadata["fps"], 2
    ) if metadata["fps"] > 0 else 0.0

    cap.release()

    logger.info(
        f"Video validated: {path.name} "
        f"({metadata['width']}x{metadata['height']}, "
        f"{metadata['fps']:.1f} FPS, "
        f"{metadata['total_frames']} frames, "
        f"{metadata['duration_sec']:.1f}s)"
    )

    return metadata


def _compute_scale_factor(width: int, height: int, max_width: int) -> float:
    """Compute scale factor to fit within max_width while preserving aspect ratio."""
    if width <= max_width:
        return 1.0
    return max_width / width


def read_frames(
    path: str,
    frame_skip: int = 1,
    max_width: int = MAX_RESOLUTION,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Read video frames as a generator.

    Args:
        path: Path to the video file.
        frame_skip: Process every Nth frame (1 = all frames).
        max_width: Auto-downscale to this width if source is larger.

    Yields:
        Tuples of (frame_index, frame_array) where frame_array is BGR numpy array.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = _compute_scale_factor(width, height, max_width)

    if scale < 1.0:
        new_w = int(width * scale)
        new_h = int(height * scale)
        logger.info(f"Auto-downscaling from {width}x{height} to {new_w}x{new_h}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                yield frame_idx, frame

            frame_idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Context manager for writing annotated video frames."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ):
        """Initialize video writer.

        Args:
            output_path: Path for the output video file.
            fps: Frames per second.
            width: Frame width in pixels.
            height: Frame height in pixels.
            codec: FourCC codec string.
        """
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer for: {output_path}")

        logger.info(f"Video writer initialized: {output_path} ({width}x{height}, {fps:.1f} FPS)")

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame."""
        self.writer.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self) -> None:
        """Release the video writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path}")
