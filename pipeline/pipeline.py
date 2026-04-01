"""Pipeline orchestration — wires detect → track → classify → aggregate → render.

This is the central orchestrator that runs the full video analysis pipeline.
Supports progress callbacks for CLI and UI integration.
"""

import csv
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from pipeline.detector import PlayerDetector
from pipeline.tracker import PlayerTracker
from pipeline.team_classifier import TeamClassifier
from pipeline.aggregator import Aggregator
from pipeline.renderer import Renderer
from utils.video_io import validate_video, read_frames, VideoWriter
from utils.schema import AppConfig, FrameResult, GameReport
from utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for progress callback: (current_frame, total_frames, stage)
ProgressCallback = Callable[[int, int, str], None]


class Pipeline:
    """Full video analysis pipeline.

    Orchestrates all stages: detection, tracking, team classification,
    aggregation, and rendering. Supports progress callbacks and
    configurable output options.
    """

    def __init__(self, config: AppConfig):
        """Initialize the pipeline with the given configuration.

        Args:
            config: Application configuration.
        """
        self.config = config

        # Initialize stages
        self.detector = PlayerDetector(
            weights=config.model.weights,
            confidence_threshold=config.model.confidence_threshold,
            device=config.model.device,
            use_field_mask=getattr(config.model, 'field_mask', True),
        )
        self.tracker = PlayerTracker(
            max_age=config.tracking.max_age,
            min_hits=config.tracking.min_hits,
            scene_change_threshold=config.tracking.scene_change_threshold,
        )
        self.classifier = TeamClassifier(
            n_clusters=config.team_classification.n_clusters,
            color_space=config.team_classification.color_space,
            sample_region=config.team_classification.sample_region,
            bootstrap_frames=config.team_classification.bootstrap_frames,
        )
        self.aggregator = Aggregator()
        self.renderer = Renderer()

        logger.info("Pipeline initialized with all stages")

    def run(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        save_video: Optional[bool] = None,
        save_json: Optional[bool] = None,
        on_progress: Optional[ProgressCallback] = None,
        train_data: Optional[Dict] = None,
    ) -> GameReport:
        """Run the full pipeline on a video file.

        Args:
            input_path: Path to the input video file.
            output_dir: Directory for output files (overrides config).
            save_video: Whether to save annotated video (overrides config).
            save_json: Whether to save JSON report (overrides config).
            on_progress: Optional callback called with (current, total, stage).

        Returns:
            GameReport with all frame results and summary.
        """
        # Resolve settings
        out_dir = output_dir or self.config.output.output_dir
        do_video = save_video if save_video is not None else self.config.output.save_video
        do_json = save_json if save_json is not None else self.config.output.save_json
        do_predictions = self.config.output.save_predictions
        frame_skip = self.config.output.frame_skip

        os.makedirs(out_dir, exist_ok=True)

        # Validate input
        if on_progress:
            on_progress(0, 0, "Validating video")
        metadata = validate_video(input_path)
        total_frames = metadata["total_frames"]
        fps = metadata["fps"]
        width = metadata["width"]
        height = metadata["height"]

        # Adjust dimensions if downscaling will occur
        from utils.video_io import MAX_RESOLUTION, _compute_scale_factor
        scale = _compute_scale_factor(width, height, MAX_RESOLUTION)
        if scale < 1.0:
            width = int(width * scale)
            height = int(height * scale)

        game_id = Path(input_path).stem
        frame_results: List[FrameResult] = []

        # Video writer setup
        video_writer = None
        if do_video:
            output_video_path = os.path.join(out_dir, "output.mp4")
            # When frame_skip > 1, only every Nth frame is written.
            # Divide FPS by frame_skip so the output plays at normal speed.
            output_fps = fps / max(frame_skip, 1)
            video_writer = VideoWriter(output_video_path, output_fps, width, height)

        # Predictions CSV setup
        predictions_file = None
        csv_writer = None
        if do_predictions:
            predictions_path = os.path.join(out_dir, "predictions.csv")
            predictions_file = open(predictions_path, "w", newline="")
            csv_writer = csv.writer(predictions_file)
            csv_writer.writerow(["frame_idx", "tracker_id", "x1", "y1", "x2", "y2", "confidence", "predicted_label"])
            logger.info(f"Predictions CSV: {predictions_path}")

        logger.info(f"Processing {input_path}: {total_frames} frames at {fps:.1f} FPS")

        team_seeds = None  # initialize before optional UI-supervised path
        if train_data:
            if on_progress:
                on_progress(0, 100, "Extracting features from UI clicks...")
            team_seeds = self._build_train_features(input_path, train_data)

        if on_progress:
            on_progress(0, total_frames, "Processing video frames")

        try:
            processed_count = 0
            blur_skipped_count = 0
            blur_threshold = getattr(self.config.output, 'blur_threshold', 0)
            for frame_idx, frame in read_frames(input_path, frame_skip=frame_skip):
                # 0. Motion blur filter — Laplacian variance measures image sharpness.
                # Blurry frames (fast camera pans) produce noisy HSV histograms that
                # pollute the K-Means bootstrap and degrade classification accuracy.
                if blur_threshold > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if blur_score < blur_threshold:
                        blur_skipped_count += 1
                        continue

                # 1. Detect
                detections = self.detector.detect(frame)

                # 2. Track
                tracked = self.tracker.update(detections, frame)

                # 3. Classify teams
                if len(tracked) > 0:
                    team_labels = self.classifier.classify(
                        frame,
                        tracked.xyxy,
                        team_seeds=team_seeds,
                        track_ids=tracked.tracker_id,
                    )
                else:
                    team_labels = []

                # 3b. Write predictions CSV
                if csv_writer is not None and len(tracked) > 0:
                    bboxes = tracked.xyxy
                    confs = tracked.confidence if hasattr(tracked, 'confidence') else [0.0] * len(tracked)
                    tids = tracked.tracker_id if tracked.tracker_id is not None else [None] * len(tracked)
                    for tid, bbox, conf, label in zip(tids, bboxes, confs, team_labels):
                        x1, y1, x2, y2 = (int(v) for v in bbox)
                        csv_writer.writerow([frame_idx, tid, x1, y1, x2, y2, f"{conf:.4f}", label])

                # 4. Aggregate
                frame_result = self.aggregator.aggregate_frame(
                    frame_id=frame_idx,
                    fps=fps,
                    detections=tracked,
                    team_labels=team_labels,
                )
                frame_results.append(frame_result)

                # 5. Render (if saving video)
                if video_writer:
                    annotated = self.renderer.annotate_frame(
                        frame=frame,
                        detections=tracked,
                        team_labels=team_labels,
                        frame_id=frame_idx,
                        team_a_count=frame_result.team_a_count,
                        team_b_count=frame_result.team_b_count,
                        other_count=frame_result.other_count,
                        confidence=frame_result.confidence,
                    )
                    video_writer.write(annotated)

                processed_count += 1
                if on_progress and processed_count % 10 == 0:
                    on_progress(processed_count, total_frames // max(frame_skip, 1), "Processing frames")

        finally:
            if video_writer:
                video_writer.release()
            if predictions_file:
                predictions_file.close()

        # Build report
        if on_progress:
            on_progress(total_frames, total_frames, "Building report")

        report = self.aggregator.build_report(
            game_id=game_id,
            total_frames=total_frames,
            fps=fps,
            frame_results=frame_results,
        )

        # Inject detected jersey colors into summary
        team_colors = self.classifier.get_team_colors()
        if team_colors:
            report.summary.team_a_color = team_colors.get("team_a", "")
            report.summary.team_b_color = team_colors.get("team_b", "")
            logger.info(
                f"Jersey colors detected: "
                f"Team A={report.summary.team_a_color}, "
                f"Team B={report.summary.team_b_color}"
            )

        # Save outputs
        if do_json:
            json_path = os.path.join(out_dir, "report.json")
            with open(json_path, "w") as f:
                f.write(report.model_dump_json(indent=2))
            logger.info(f"JSON report saved: {json_path}")

            summary_path = os.path.join(out_dir, "summary.txt")
            with open(summary_path, "w") as f:
                f.write(report.to_summary_text())
            logger.info(f"Summary saved: {summary_path}")

        if on_progress:
            on_progress(total_frames, total_frames, "Complete")

        logger.info(
            f"Pipeline complete: {len(frame_results)} frames processed"
            + (f", {blur_skipped_count} blurry frames skipped ({blur_skipped_count / max(processed_count + blur_skipped_count, 1):.0%}) with blur_threshold={blur_threshold}" if blur_threshold > 0 else "")
            + f", avg_confidence={report.summary.avg_confidence}"
        )

        return report

    def _build_train_features(self, video_path: str, train_data: Dict) -> Dict[str, np.ndarray]:
        import cv2
        import numpy as np
        from collections import defaultdict
        
        team_seeds = {"team_a": [], "team_b": [], "other": []}
        
        # Group clicks by frame to avoid seeking back and forth
        clicks_by_frame = defaultdict(list)
        for team, clicks in train_data.items():
            if not isinstance(clicks, list):
                continue
            for c in clicks:
                if "frame" in c and "x" in c and "y" in c:
                    clicks_by_frame[c["frame"]].append({"team": team, "x": c["x"], "y": c["y"]})
                    
        if not clicks_by_frame:
            return None
                
        cap = cv2.VideoCapture(str(video_path))
        for f_idx, points in clicks_by_frame.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Run detector on this frame
            detections = self.detector.detect(frame)
            bboxes = detections.xyxy if hasattr(detections, "xyxy") else []
            
            for pt in points:
                px, py = pt["x"], pt["y"]
                # Find which bbox contains this point
                matched_bbox = None
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        matched_bbox = bbox
                        break
                        
                if matched_bbox is None:
                    # Fallback option B: draw a 50x50 box around the click
                    matched_bbox = np.array([px-25, py-25, px+25, py+25])
                
                # Extract features using exactly the same logic
                feat = self.classifier._extract_color_features(frame, matched_bbox)
                if feat is not None:
                    team_seeds[pt["team"]].append(feat)
                    
        cap.release()
        
        # Convert to numpy arrays
        for k in list(team_seeds.keys()):
            if len(team_seeds[k]) > 0:
                team_seeds[k] = np.array(team_seeds[k])
            else:
                team_seeds[k] = np.array([])
            
        return team_seeds
