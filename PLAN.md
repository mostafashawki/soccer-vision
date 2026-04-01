# soccer-vision — Project Plan

> **Senior ML Engineer Challenge — Use Case 2: Sports Analytics**
> Customer B wants to analyze soccer video footage to count how many players
> per team are visible on screen at any given moment.

---

## Goal

Build a production-ready video analytics pipeline that processes soccer match
footage and outputs per-frame player counts per team, with confidence scores
and a structured JSON report. The system should be immediately usable by a
non-technical customer and extensible toward real-time and large-scale
deployment.

---

## Scope for Implementation

> **Tier 1 only.** Tiers 2 and 3 are architecture decisions documented below
> for roadmap and presentation purposes — do not implement them.

---

## Input / Output Contract

**Input:**
- A soccer match video file (`.mp4`, `.avi`, `.mov`)
- Optional config overrides via `config.yaml`

**Output:**
- `report.json` — structured per-frame analytics
- `output.mp4` — annotated video with bounding boxes, team labels, and live counter overlay
- `summary.txt` — human-readable summary for non-technical stakeholders (auto-generated from JSON report)

**Report schema:**
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

> **Note:** `confidence` is defined as the average detection confidence
> (from YOLOv8) across all detected persons in that frame.

---

## System Components (Tier 1)

```
soccer-vision/
├── config.yaml               # All tunable parameters
├── main.py                   # CLI entrypoint
├── api/
│   ├── __init__.py
│   └── server.py             # FastAPI — POST /process, GET /status, GET /results
├── ui/
│   └── app.py                # Streamlit — upload video, progress bar, results viewer
├── pipeline/
│   ├── __init__.py
│   ├── detector.py           # YOLOv8 person detection
│   ├── tracker.py            # ByteTrack multi-object tracking (via supervision)
│   ├── team_classifier.py    # Jersey color clustering (K-means, HSV)
│   ├── aggregator.py         # Per-frame count aggregation + confidence
│   └── renderer.py           # Annotated video + overlay generation (via supervision)
├── utils/
│   ├── video_io.py           # Video read/write, frame extraction, validation
│   ├── logger.py             # Structured logging (not print statements)
│   └── schema.py             # Pydantic output models
├── scripts/
│   └── download_sample.sh    # Downloads sample video from GCS for testing
├── tests/
│   ├── test_detector.py
│   ├── test_team_classifier.py
│   └── test_aggregator.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── PLAN.md                   # This file
```

---

## Tech Stack & Rationale

| Component | Choice | Why |
|---|---|---|
| Detection | YOLOv8n (ultralytics) | Pretrained on COCO, persons class works out of the box — no training needed |
| Tracking | ByteTrack (via `supervision`) | Handles occlusion well, lightweight, built-in to supervision — minimal glue code |
| Annotation | `supervision` (Roboflow) | Pre-built annotators for bounding boxes, labels, and overlays — eliminates manual OpenCV drawing |
| Team classification | K-means on HSV jersey colors | Zero-label, elegant, fast — avoids unnecessary model complexity |
| Video I/O | OpenCV | Industry standard, handles codec variety |
| Output validation | Pydantic | Enforces output contract, catches bugs early |
| Logging | Python `logging` + JSON formatter | Machine-readable logs, production habit |
| Config | PyYAML + dataclass | Single source of truth for all parameters |
| API | FastAPI | Thin wrapper around the pipeline; enables UI integration and future Tier 2 expansion |
| UI | Streamlit | Drag-and-drop upload, progress bar, video playback, charts — zero frontend build system |
| Containerization | Docker + Docker Compose | `docker compose up` runs the full stack — API + UI |

---

## Configuration (`config.yaml`)

```yaml
model:
  weights: yolov8n.pt
  confidence_threshold: 0.5
  device: auto               # auto-selects GPU if available, else CPU

tracking:
  max_age: 30                # frames before track is dropped
  min_hits: 3                # frames before track is confirmed
  scene_change_threshold: 0.8  # normalized frame diff to trigger tracker reset

team_classification:
  n_clusters: 3              # team_a, team_b, other (goalkeeper/referee)
  color_space: HSV
  sample_region: torso       # crop bounding box to torso for jersey color

output:
  output_dir: ./output       # directory for all generated files
  save_video: true
  save_json: true
  frame_skip: 1              # process every Nth frame (1 = all frames)
  log_level: INFO
```

---

## Edge Cases & Mitigations

| Edge Case | Mitigation |
|---|---|
| Goalkeeper / referee jersey confusion | Max-distance pair selection + confidence gate; labeled as `other` |
| Sun/shade splitting one team into 2 clusters | Post-selection cluster merging: orphan clusters close to a team centroid are merged into that team |
| Occlusion (players overlapping) | ByteTrack maintains track identity through short occlusions |
| Camera cuts & replays | Scene change detection via frame difference threshold; tracker resets |
| Motion blur / low quality frames | Per-frame confidence score; frames below threshold are flagged |
| Wrong codec / corrupted video | Input validation in `video_io.py` with clear error messages |
| > 4K resolution | Auto-downscale to 1080p before inference to maintain performance |
| Large file upload denial of service | API enforces 500 MB upload limit |

---

## CLI Interface

```bash
# Basic usage
python main.py --input sample.mp4

# With custom config and output directory
python main.py --input sample.mp4 --config config.yaml --output-dir ./results

# Skip video output (faster, report only)
python main.py --input sample.mp4 --no-video

# Docker (CLI mode)
docker run -v $(pwd)/data:/data soccer-vision --input /data/sample.mp4
```

---

## Web Interface (Streamlit + FastAPI)

The web interface provides a browser-based workflow for non-technical users:

1. **Upload** — Drag-and-drop a video file in the Streamlit UI
2. **Process** — Click "Analyze" → FastAPI triggers the pipeline with real-time progress
3. **Review** — View annotated video, per-frame charts, and download JSON report

```bash
# Start the full stack (API + UI)
docker compose up

# Open http://localhost:8501 in a browser
```

### FastAPI Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/process` | Upload video and start processing |
| `GET` | `/status/{job_id}` | Check processing progress |
| `GET` | `/results/{job_id}` | Download results (JSON + video) |
| `GET` | `/health` | Health check |

> **Note:** This remains a synchronous, single-job setup in Tier 1.
> No job queue or async workers — that's Tier 2.

---

## Sample Video

A sample soccer clip is provided for testing:

```bash
bash scripts/download_sample.sh
# Downloads to data/sample.mp4 (~XX MB)
# Source: https://storage.googleapis.com/ml6_senior_ml_engineer_challenge/sample.mp4
```

---

## Pipeline Progress Reporting

The pipeline exposes a progress callback mechanism so both the CLI and Streamlit UI
can report real-time progress:

```python
# Pipeline yields (current_frame, total_frames, stage) as it processes
for progress in pipeline.run(video_path, on_progress=callback):
    # CLI: print progress bar
    # Streamlit: update st.progress()
    # FastAPI: update job status
```

---

## Out of Scope (Tier 1)

Explicitly excluded from implementation — documented for roadmap only:

- Job queue / async processing
- Database persistence
- Authentication / authorization
- Multi-video batch processing
- Real-time / live stream support
- Cloud deployment (S3, GCS, Lambda, etc.)
- Heatmaps, formation detection, pressing intensity
- Model retraining or fine-tuning pipeline
- Monitoring / alerting infrastructure

---

## Architecture Roadmap (Presentation Reference)

### Tier 2 — Growth (10–100 games/day)

```
Client → REST API (FastAPI)
             ↓
         Job Queue (Celery + Redis)
             ↓
      Worker Pool (auto-scaling GPU)
             ↓
      Object Storage (S3/MinIO)    ← processed video + JSON report
             ↓
      PostgreSQL (job metadata)
             ↓
      Webhook / polling → Client
```

**Key features:**
- Async job submission: `POST /jobs` → returns `job_id`
- Status polling: `GET /jobs/{job_id}` with percentage progress
- GPU workers scale based on queue depth (Celery + auto-scaling)
- Results cached by video SHA-256 hash (idempotent reprocessing)
- Estimated cost: ~$0.05–0.10 per game at GPU spot pricing

**Additional capabilities (Tier 2):**

| Feature | Description |
|---|---|
| Webhook notifications | `POST` results to customer-provided callback URL on job completion |
| Video deduplication | SHA-256 hash of input → skip processing if identical video exists |
| Result persistence | PostgreSQL for job metadata, S3/MinIO for artifacts (video + JSON) |
| Multi-tenant support | API key per customer, results scoped by tenant — foundation for SaaS |
| Batch upload | `POST /jobs/batch` — accepts a list of video URLs, queued individually |
| Health & metrics | `/health` endpoint + Prometheus metrics (`jobs_processed`, `avg_latency`, `gpu_util`) |
| CI/CD pipeline | GitHub Actions → build image → push to ECR → deploy to ECS/EKS |
| Model A/B testing | Route a percentage of jobs to a new model version; compare confidence distributions |

### Tier 3 — Scale (1000+ games/day, live broadcast)

```
RTMP/HLS Stream → Kafka → Flink (stream processing)
                               ↓
                    Triton Inference Server (GPU cluster)
                               ↓
                    Redis (frame-level cache)
                               ↓
                    WebSocket API / CDN → Broadcast overlay / consumers
```

**Key features:**
- Sub-second latency SLA for live broadcast overlay
- Model served via NVIDIA Triton for batched GPU inference
- Horizontal scaling with Kubernetes (multi-node GPU cluster)
- Multi-region deployment for global

### Improved Accuracy & Supervised Classification (Fix 3)
- **The "Data Engine" Approach**: Instead of beginning with manual human labeling (which suffers from a cold-start, is slow, and expensive), we rely on robust mathematical edge-case filters in Tier 1 (`Unsupervised K-means Dominant Color Extraction` and `HSV Pitch Masking`). These act as an **Auto-Labeler** for generating massive pre-annotated datasets overnight.
- **Supervised Team Classifier**: Once the data is generated, humans only need to review the remaining ~5-10% of errors to correct the edge cases. We then replace the unsupervised K-means with a trained MobileNet/ResNet model on the reviewed dataset. By training on these complex edge-cases, we eliminate the need for the user to seed colors per-game, all without the effort of annotating 50,000 frames from scratch.
- **Advanced Tracking**: Evaluate BoT-SORT for better occlusion handling if ByteTrack drops IDs.
- **Pitch Registration**: Implement homography to map pixel coordinates to a 2D top-down pitch view.|

**Additional capabilities (Tier 3):**

| Feature | Description |
|---|---|
| RTMP/HLS live ingest | Accept live broadcast streams, process in near-real-time, output via WebSocket |
| Edge inference | Quantized YOLO (TensorRT/ONNX) on stadium hardware for sub-frame latency |
| Multi-camera fusion | Stitch views from multiple angles; homography-based coordinate mapping for full pitch |
| Player re-identification | Deep re-ID model (OSNet) to maintain identity across camera cuts and close-ups |
| Formation detection | Voronoi tessellation or GNN-based classifier from player coordinates |
| Tactical analytics API | Pressing intensity, possession zones, pass networks — coaching dashboard integration |
| Data flywheel | Customer corrections → retraining labels; auto-improve model per team/league over time |
| Compliance engine | Auto-blur faces for non-consented footage; GDPR/CCPA data retention automation |

---

## MLOps Considerations (Presentation Reference)

| Concern | Approach |
|---|---|
| Model versioning | MLflow or S3-versioned model artifacts |
| Drift detection | Monitor confidence score distribution over time; alert on degradation |
| Retraining trigger | Seasonal (new jerseys), or when confidence drops below threshold |
| Evaluation dataset | 50–100 manually annotated frames per new season |
| Experiment tracking | MLflow or Weights & Biases |

---

## Team Kickoff Plan (3-person team)

**Week 1–2: Foundation**
- Junior: Set up repo, Docker, data pipeline, video I/O module
- Senior: Validate detection + tracking quality on sample video, define evaluation metrics
- PM: Align with customer on acceptance criteria, define what "correct" means

**Month 1: Working Prototype**
- Junior: Build aggregator, renderer, output schema
- Senior: Team classifier, edge case handling, confidence scoring
- PM: Weekly customer demos from day 1 — no waterfall

**Month 2: Hardening**
- Junior: Unit tests, CI pipeline, README documentation
- Senior: Performance optimization, config system, Docker packaging
- PM: Collect customer feedback, prioritize v2 features

**Month 3+: Expansion**
- REST API layer (Tier 2 foundation)
- Heatmap feature (first product extension)
- Roadmap review with customer

> **Key principle:** Ship a working demo to the customer by end of Week 2.
> Feedback loops beat perfect planning every time.

---

## Business Value & Customer Expansion

| Phase | Feature | Customer Segment |
|---|---|---|
| MVP | Player count per team | Club analysts, broadcast |
| v2 | Player heatmaps | Coaching staff, scouting |
| v3 | Formation detection | Tactical analysis platforms |
| v4 | Pressing intensity score | Live broadcast overlay |
| v5 | Multi-camera fusion | Full-pitch coverage, VAR support |

**Revenue model options:**
- Per-video processing fee (pay-as-you-go)
- Monthly SaaS subscription per club
- White-label API for broadcast partners

---

## Security & Compliance Notes

- Videos are proprietary customer data — never store beyond agreed retention window
- GDPR: player tracking data may constitute biometric data in some jurisdictions — legal review required before production
- API authentication via JWT if exposed externally (Tier 2+)
- All data in transit encrypted (TLS 1.3)

---

## Definition of Done (Tier 1)

- [ ] Processes `sample.mp4` end-to-end without errors
- [ ] Outputs valid `report.json` matching defined schema
- [ ] Outputs annotated `output.mp4` with bounding boxes and team counter
- [ ] Runs inside Docker with `docker compose up`
- [ ] Streamlit UI allows video upload, shows progress, and displays results
- [ ] FastAPI endpoints respond correctly (`/process`, `/status`, `/results`, `/health`)
- [ ] Unit tests pass for detector, classifier, and aggregator
- [ ] README explains setup and usage clearly enough for a junior engineer
- [ ] Config is fully externalized — no hardcoded values in code
- [ ] Logging is structured and informative (not `print()`)
- [ ] Sample video downloadable via `scripts/download_sample.sh`
- [ ] Unsupervised Field Masking implemented
- [ ] Semi-supervised UI Seeding implemented

---

## Implementation Tasks (Tier 1)

> Each task is a self-contained unit of work resulting in a **single atomic commit**.
> Tasks are ordered by dependency — each builds on the previous.

---

### Task 1 — Project Scaffolding

**Scope:** Create the folder structure, config, and dependency files.

**Deliverables:**
- Directory structure (`pipeline/`, `api/`, `ui/`, `utils/`, `scripts/`, `tests/`)
- `__init__.py` files
- `config.yaml` with all default parameters
- `requirements.txt` with pinned dependencies
- `.gitignore` (Python defaults + data/ + output/ + model weights)
- `.dockerignore`

**Commit:** `feat: project scaffolding and configuration`

---

### Task 2 — Core Utilities

**Scope:** Build the foundational modules that all pipeline stages depend on.

**Deliverables:**
- `utils/logger.py` — structured JSON logger with configurable level
- `utils/video_io.py` — video reader (frame generator), writer, input validation, auto-downscale
- `utils/schema.py` — Pydantic models for `FrameResult`, `GameReport`, `Summary`

**Commit:** `feat: core utilities — logger, video I/O, Pydantic schemas`

---

### Task 3 — Player Detection Module

**Scope:** YOLOv8 person detection with configurable confidence threshold.

**Deliverables:**
- `pipeline/detector.py` — loads YOLOv8n, filters for person class (class 0), returns `sv.Detections`
- `tests/test_detector.py` — unit tests with a sample frame

**Commit:** `feat: YOLOv8 person detection module`

---

### Task 4 — Multi-Object Tracking

**Scope:** ByteTrack tracking via supervision to maintain player identity across frames.

**Deliverables:**
- `pipeline/tracker.py` — wraps `sv.ByteTrack`, handles tracker reset on scene changes
- Scene change detection via frame difference threshold

**Commit:** `feat: ByteTrack multi-object tracking with scene change detection`

---

### Task 5: Team Classification (KNN Multi-Frame Seeding)
- **Goal**: Group players into 2 teams + referee/others based on jersey color.
- **Why**: Essential for team-specific analytics. Shirts look different from the back vs front, so a supervised approach is needed.
- **How**:
  - Extract torso region (middle 40% of bounding box) to avoid shorts/socks.
  - Convert BGR to HSV color space.
  - Run **K-Nearest Neighbors (KNN)** using a dynamically generated training set from the user's UI clicks.
- **Improvements added in development**:
  - **Fix 1 (Unsupervised Field Mask)**: Added a pitch-green HSV mask filter inside the detector. Detections whose bottom-center (feet) don't fall on the green grass are discarded, effectively stripping out spectators and coaches.
  - **Fix 2 (Semi-Supervised Multi-Frame UI)**: Replaced 1-frame K-means with an interactive 5-frame Streamlit labeling wizard. The user can click multiple examples of Team A, Team B, and Referees across different frames. The pipeline takes these clicks, trains a local `KNeighborsClassifier`, and accurately classifies the entire video.
- **Deliverable**: `pipeline/team_classifier.py` and Streamlit UI.

**Commit:** `feat: K-means jersey color team classification`

---

### Task 6 — Aggregation & Confidence Scoring

**Scope:** Combine detection + tracking + classification into per-frame counts.

**Deliverables:**
- `pipeline/aggregator.py` — produces `FrameResult` per frame, computes confidence score
- Generates `GameReport` with summary statistics (averages)
- `tests/test_aggregator.py` — unit tests with mock frame data

**Commit:** `feat: per-frame aggregation and confidence scoring`

---

### Task 7 — Annotated Video Renderer

**Scope:** Draw bounding boxes, team labels, and live counter overlay on video frames.

**Deliverables:**
- `pipeline/renderer.py` — uses `sv.BoxAnnotator`, `sv.LabelAnnotator`, plus custom counter overlay
- Writes annotated `output.mp4` via `utils/video_io.py`

**Commit:** `feat: annotated video renderer with team overlay`

---

### Task 8 — Pipeline Orchestration & CLI

**Scope:** Wire all pipeline stages together and expose via CLI.

**Deliverables:**
- `pipeline/__init__.py` — `Pipeline` class orchestrating detect → track → classify → aggregate → render
- Progress callback mechanism (`on_progress(current_frame, total_frames)`)
- `main.py` — argparse CLI with `--input`, `--config`, `--output-dir`, `--no-video`
- Writes `report.json` and `summary.txt` to output directory

**Commit:** `feat: pipeline orchestration and CLI entrypoint`

---

### Task 9 — FastAPI Server

**Scope:** Thin API layer wrapping the pipeline for programmatic and UI access.

**Deliverables:**
- `api/server.py` — `POST /process`, `GET /status/{job_id}`, `GET /results/{job_id}`, `GET /health`
- Synchronous processing (Tier 1 — no background workers)
- File upload handling, result file serving

**Commit:** `feat: FastAPI server with processing endpoints`

---

### Task 10 — Streamlit UI

**Scope:** Browser-based interface for non-technical users.

**Deliverables:**
- `ui/app.py` — video upload, config panel, progress bar, results viewer
- Displays annotated video, per-frame team count chart, and downloadable JSON report
- Calls FastAPI endpoints under the hood

**Commit:** `feat: Streamlit UI with upload, progress, and results`

---

### Task 11 — Docker & Sample Video

**Scope:** Containerize the full stack and provide a sample video for testing.

**Deliverables:**
- `Dockerfile` — multi-stage build, Python 3.11, installs dependencies, copies source
- `docker-compose.yml` — `api` and `ui` services, volume mounts for data/output
- `scripts/download_sample.sh` — downloads sample video from GCS
- Validates `docker compose up` runs the full pipeline end-to-end

**Commit:** `feat: Docker containerization and sample video script`

---

### Task 12 — README & Final Polish

**Scope:** Documentation and cleanup.

**Deliverables:**
- `README.md` — project overview, quickstart, architecture, configuration reference, troubleshooting
- Review all logging, error messages, and edge case handling
- Verify Definition of Done checklist passes

**Commit:** `docs: README and final polish`