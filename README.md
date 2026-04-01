# ⚽ Soccer Vision

**AI-powered player counting per team from soccer match footage.**

**89.4% per-track accuracy — zero labeling, zero GPU, pure OpenCV math (Tier 1 unsupervised)**

Soccer Vision processes soccer video footage and outputs per-frame player counts per team, with confidence scores, annotated video, and a structured JSON report.

---

## Features

- **Player Detection** — YOLOv8n pretrained on COCO (no training required)
- **Field Pitch Masking** — Unsupervised HSV green mask filters out spectators and coaches outside the field
- **Motion Blur Frame Filtering** — Laplacian variance gate (`blur_threshold` in config) automatically skips motion-blurred frames caused by fast camera pans before they reach the detector, protecting the K-Means bootstrap from noisy feature vectors that would degrade classification accuracy for the entire video
- **Multi-Object Tracking** — ByteTrack via [supervision](https://github.com/roboflow/supervision) for identity persistence across frames
- **Team Classification** — Two-phase pipeline: unsupervised bootstrap (first 50 frames) builds a frozen KNN classifier, used for all subsequent frames — no per-frame re-clustering
- **HSV Histogram Features** — 18-dimensional normalized histogram (16 hue bins + white fraction + dark fraction) per jersey crop; far more discriminative than a single Lab color average
- **Max-Distance Team Pairing** — The two cluster centers furthest apart in feature space are selected as the two teams, making the mapping robust to referees and goalkeepers regardless of how often they appear in the bootstrap window
- **Distance Confidence Gate** — Players whose feature vector exceeds min(50% of team–team separation, 0.8) from the nearest team centroid are reclassified as "other", preventing referee/kit-edge-case bleed while avoiding over-rejection for teams with similar colors
- **Cluster Merging** — After max-distance team selection, orphan K-Means clusters that are close to a team centroid are merged into that team, handling sun/shade splits that would otherwise misclassify half a team as "other"
- **Track-Level Majority Vote** — A 15-frame rolling vote per `tracker_id` eliminates per-frame label flicker caused by occlusion or lighting changes
- **Semi-Supervised UI Seeding** — Streamlit UI allows users to click players on frame 1 to seed Team A and Team B precisely, bypassing the bootstrap phase entirely
- **Annotated Video** — Bounding boxes, team labels, and live counter overlay
- **Structured Report** — JSON report with per-frame counts and confidence scores
- **Web UI** — Streamlit dashboard for drag-and-drop video analysis using `streamlit-image-coordinates`
- **REST API** — FastAPI endpoints for programmatic access (500 MB upload limit)
- **Docker** — Single `docker compose up` to run the full stack

---

## Quick Start

### 1. Clone & Download Sample Video

```bash
git clone https://github.com/mostafashawki/soccer-vision.git
cd soccer-vision
bash scripts/download_sample.sh
```

### 2. Run with Docker (Recommended)

```bash
docker compose up --build
```

- **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs**: [http://localhost:8004/docs](http://localhost:8004/docs)
- **Labeling Tool**: [http://localhost:8502](http://localhost:8502) *(available after processing a video)*

Upload `data/sample.mp4` in the Streamlit UI and click "Analyze Video".

### 3. Run Locally (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# CLI usage
python main.py --input data/sample.mp4

# Or start the API + UI
uvicorn api.server:app --host 0.0.0.0 --port 8000 &
streamlit run ui/app.py
```

---

## CLI Usage

```bash
# Basic usage
python main.py --input data/sample.mp4

# Custom config and output directory
python main.py --input data/sample.mp4 --config config.yaml --output-dir ./results

# Skip video output (faster, report only)
python main.py --input data/sample.mp4 --no-video
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process` | Upload video and start processing |
| `GET` | `/status/{job_id}` | Check processing progress |
| `GET` | `/results/{job_id}/report` | Download JSON report |
| `GET` | `/results/{job_id}/video` | Download annotated video |
| `GET` | `/results/{job_id}/summary` | Get human-readable summary |
| `GET` | `/health` | Health check |

Interactive API docs available at [http://localhost:8004/docs](http://localhost:8004/docs).

---

## Output

### report.json

```json
{
  "game_id": "sample",
  "processed_at": "2026-03-28T10:00:00Z",
  "total_frames": 1800,
  "fps": 30,
  "frames": [
    {
      "frame_id": 42,
      "timestamp_sec": 1.4,
      "team_a_count": 4,
      "team_b_count": 3,
      "other_count": 1,
      "confidence": 0.91
    }
  ],
  "summary": {
    "avg_team_a": 4.2,
    "avg_team_b": 3.8,
    "avg_confidence": 0.88
  }
}
```

### output.mp4

Annotated video with:
- Color-coded bounding boxes per team
- Team labels with confidence scores
- Live counter overlay (top-left corner)

### summary.txt

Human-readable summary for non-technical stakeholders.

---

## Configuration

All parameters are externalized in `config.yaml`:

```yaml
model:
  weights: yolov8n.pt           # YOLO model weights
  confidence_threshold: 0.5     # Detection confidence threshold
  device: auto                  # auto / cpu / cuda

tracking:
  max_age: 30                   # Frames before track is dropped
  min_hits: 3                   # Frames before track is confirmed
  scene_change_threshold: 0.8   # Frame diff threshold for camera cut detection

team_classification:
  n_clusters: 4                 # K-means clusters (team_a, team_b, + 2 other slots for GK/referee)
  bootstrap_frames: 50          # Frames sampled before freezing the KNN classifier
  color_space: HSV              # Color space for feature extraction
  sample_region: torso          # Crop region (0.25–0.55 of bbox height, min 40px)

output:
  output_dir: ./output          # Output directory
  save_video: true              # Generate annotated video
  save_json: true               # Generate JSON report
  frame_skip: 1                 # Process every Nth frame
  blur_threshold: 100           # Skip blurry frames (Laplacian variance < threshold)
                                # 0=disabled | 50=loose | 100=balanced | 200=strict
  log_level: INFO               # Logging level
```

---

## Architecture

```
Input Video
    ↓
Motion Blur Filter (Laplacian variance gate — skips blurry frames before detection)
    ↓
Detection (YOLOv8) + Field Pitch Mask (HSV green filter)
    ↓
Tracking (ByteTrack) — persistent tracker_id per player
    ↓
Team Classification
    ├─ Bootstrap phase (frames 0–49)
    │     Extract 18D HSV histogram per torso crop
    │     Batch K-Means (n_clusters=4) over all collected features
    │     Select 2 cluster centers with max histogram distance → team_a / team_b
    │     Merge orphan clusters close to a team centroid
    │     Freeze result as KNN classifier
    │
    └─ Processing phase (frame 50+)
          KNN.predict() per crop
          Distance confidence gate → reassign uncertain predictions to "other"
          Track-level 15-frame majority vote per tracker_id
    ↓
Aggregation (per-frame counts)
    ↓
          ┌────────────────────┬──────────────────┐
          ↓                    ↓                  ↓
    report.json          output.mp4         summary.txt
```

---

## Future Roadmap

### The "Data Engine" Approach to Labeling
*Why don't we just start with human labeling (supervised learning) right away?*
As a best practice in modern ML engineering, we delay slow and expensive manual human annotation by building strong mathematical filters **first**. The bootstrap + HSV histogram pipeline solves ~85–90% of cases automatically and can act as an **auto-labeler**: run it overnight to generate pre-annotated crops, then have humans correct only the edge-case errors — reducing labeling effort by 10x.

The next step toward crossing 95%+ accuracy is **few-shot supervised seeding**: label ~20 jersey crops per team (once per kit combination), use those confirmed examples to override the bootstrap KNN seeds, and the rest of the pipeline remains unchanged.

**Tier 2 — Labeling & Supervised Classification**
1. Export per-frame predictions to CSV (`frame_id`, `tracker_id`, `predicted_label`) and label ~100 bounding boxes in [CVAT](https://cvat.org) or [Label Studio](https://labelstud.io) (import existing detections as pre-annotations, only correct wrong labels)
2. Compute per-frame accuracy, per-track accuracy, and flip rate to establish a baseline metric before and after each algorithmic change
3. Replace bootstrap KNN seeds with few-shot labeled jersey crops (~20 per team); this alone is expected to push accuracy above 95%
4. Optionally replace the KNN with a lightweight CNN (MobileNet/ResNet) trained via [Roboflow](https://roboflow.com) for full supervised classification

**Tier 3 — Infrastructure & Tracking**
1. **Database & Message Queue** — PostgreSQL + Celery/Redis for multi-job architecture instead of synchronous FastAPI blocking
2. **Advanced Tracking** — Evaluate BoT-SORT for robust occlusion handling when players intersect
3. **Jersey Segmentation** — Replace fixed torso crop with SAM2 segmentation to remove arm/ball/background pixels from feature extraction

---

## Project Structure

```
soccer-vision/
├── config.yaml               # Configuration
├── main.py                   # CLI entrypoint
├── api/
│   └── server.py             # FastAPI server
├── ui/
│   └── app.py                # Streamlit UI
├── pipeline/
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── detector.py           # YOLOv8 person detection
│   ├── tracker.py            # ByteTrack tracking
│   ├── team_classifier.py    # Jersey color classification
│   ├── aggregator.py         # Count aggregation
│   └── renderer.py           # Video annotation
├── utils/
│   ├── video_io.py           # Video I/O utilities
│   ├── logger.py             # Structured logging
│   └── schema.py             # Pydantic models
├── scripts/
│   ├── download_sample.sh    # Sample video download
│   ├── label_tracks.py       # Streamlit labeling tool (ground truth)
│   └── evaluate.py           # Evaluation metrics (accuracy, F1, flip rate)
├── labels/                   # Ground truth CSVs (git-ignored, created by label_tracks.py)
├── tests/                    # Unit tests
├── Dockerfile
├── docker-compose.yml
└── PLAN.md                   # Project plan & architecture
```

---

## Evaluation Pipeline

The evaluation pipeline measures team classification accuracy by comparing model predictions against human-corrected ground truth labels. It's a 3-step process:

### Step 1 — Process a video (generate predictions)

Ensure `save_predictions: true` is set in `config.yaml` (default), then process a video:

```bash
# Via Docker (recommended)
docker compose up --build -d
# Upload data/sample.mp4 via the Streamlit UI at http://localhost:8501
# Or via CLI:
docker compose exec api python main.py --input data/sample.mp4
```

This generates `output/predictions.csv` with columns:
```
frame_idx, tracker_id, x1, y1, x2, y2, confidence, predicted_label
```

### Step 2 — Create ground truth labels (human correction)

The labeling tool shows one crop per tracked player with the model's prediction pre-filled. You only correct the wrong ones — typically ~15-20% of tracks.

```bash
# The labeling tool starts automatically with docker compose up
# Open http://localhost:8502
```

Open [http://localhost:8502](http://localhost:8502), review/correct labels in the grid, then click **Save Labels**. This produces `labels/ground_truth.csv`:
```
tracker_id,true_label
3,team_a
7,team_b
12,other
```

### Step 3 — Run evaluation

```bash
# Via Docker
docker compose run --rm \
  -v ./labels:/app/labels \
  api python scripts/evaluate.py \
    --predictions output/predictions.csv \
    --ground-truth labels/ground_truth.csv

# Or locally
python scripts/evaluate.py \
    --predictions output/predictions.csv \
    --ground-truth labels/ground_truth.csv
```

Sample output:
```
============================================================
  SOCCER VISION — EVALUATION REPORT
============================================================
  Predictions :  output/predictions.csv
  Ground truth:  labels/ground_truth.csv
  Tracks labeled: 24
  Detection rows: 4320

────────────────────────────────────────────────────────────
  CORE METRICS
────────────────────────────────────────────────────────────
  Per-detection accuracy       85.2%  [#################---]  Acceptable
  Per-track accuracy           91.7%  [##################--]  Good
  Flip rate                     2.1%  [####################]  Good

  Benchmark targets:
  Metric                       Acceptable      Good   Production
                                  > 80%    > 90%       > 95%  per-detection
                                  > 85%    > 95%       > 98%  per-track
                                  <  5%    <  2%      < 0.5%  flip rate
```

### Benchmark Targets

| Metric | Acceptable | Good | Production |
|--------|-----------|------|------------|
| Per-detection accuracy | > 80% | > 90% | > 95% |
| Per-track accuracy | > 85% | > 95% | > 98% |
| Flip rate | < 5% | < 2% | < 0.5% |

The unsupervised HSV bootstrap (Tier 1) typically achieves "Acceptable" to "Good" range. Crossing into "Production" requires Tier 2 supervised fine-tuning (see [Future Roadmap](#future-roadmap)).

---

## Testing

```bash
# Via Docker (recommended)
docker compose run --rm \
  -v ./tests:/app/tests \
  api pytest tests/ -v

# Or locally
pip install -r requirements.txt
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8n (ultralytics) |
| Tracking | ByteTrack (supervision) |
| Annotation | supervision |
| Team Classification | K-means (scikit-learn) |
| Video I/O | OpenCV |
| Schema | Pydantic |
| API | FastAPI |
| UI | Streamlit |
| Container | Docker + Docker Compose |

---

## License

This project was built as part of a technical challenge. All rights reserved.
