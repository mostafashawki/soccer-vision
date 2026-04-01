"""soccer-vision CLI entrypoint.

Usage:
    python main.py --input sample.mp4
    python main.py --input sample.mp4 --config config.yaml --output-dir ./results
    python main.py --input sample.mp4 --no-video
"""

import argparse
import sys
import os

from utils.schema import AppConfig
from utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Soccer Vision — Video analytics pipeline for player counting per team",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/sample.mp4
  python main.py --input data/sample.mp4 --config config.yaml
  python main.py --input data/sample.mp4 --output-dir ./results --no-video
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input video file (.mp4, .avi, .mov)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (overrides config value)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video output (faster, JSON report only)",
    )
    return parser.parse_args()


def cli_progress(current: int, total: int, stage: str) -> None:
    """Print progress to stdout."""
    if total > 0:
        pct = min(100, int(current / total * 100))
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  {stage}: [{bar}] {pct}% ({current}/{total})", end="", flush=True)
        if current >= total:
            print()
    else:
        print(f"  {stage}...")


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    # Load config
    if os.path.exists(args.config):
        config = AppConfig.from_yaml(args.config)
    else:
        config = AppConfig()

    logger = get_logger("soccer-vision", level=config.output.log_level)
    logger.info(f"Starting soccer-vision pipeline")
    logger.info(f"Input: {args.input}")

    # Import pipeline here to defer heavy model loading
    from pipeline import Pipeline

    pipeline = Pipeline(config)

    try:
        report = pipeline.run(
            input_path=args.input,
            output_dir=args.output_dir,
            save_video=not args.no_video,
            on_progress=cli_progress,
        )

        print(f"\n{'=' * 50}")
        print(report.to_summary_text())
        print(f"{'=' * 50}")
        print(f"\nOutput saved to: {args.output_dir or config.output.output_dir}")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
