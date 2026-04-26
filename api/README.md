# LENS+ Python API

This service handles WebRTC signaling, receives camera frames, throttles analysis to a fixed FPS, and sends inference-style messages over a WebRTC data channel.

It is intentionally lightweight so you can swap in a real model pipeline with minimal changes.

## What this API does

- Accepts SDP offer/answer exchange (`POST /webrtc/offer`)
- Accepts trickle ICE candidates (`POST /webrtc/ice`)
- Tracks active sessions and frame counters (`GET /debug/sessions`)
- Returns persisted session dump history (`GET /debug/sessions/history`)
- Exposes latest JPEG snapshot per session (`GET /debug/sessions/{session_id}/latest.jpg`)
- Writes all processed session frames to disk (`api/app/session_artifacts/` by default)
- Sends mock inference payloads on data channel `results` (replace this with your model output)

Core implementation is in `api/app/main.py`.

## Run locally

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```text
http://localhost:8000/health
```

## Environment variables

- `ANALYSIS_TARGET_FPS` (default: `15`)
  - Controls server-side frame processing rate.
  - This is the only source of truth for analysis cadence.
  - Value is clamped to `1..30`.
- `SESSION_ARTIFACTS_DIR` (optional)
  - Directory where per-session frame dumps are written.
  - Defaults to `api/app/session_artifacts` in local runs and `/app/app/session_artifacts` in Docker.

Examples:

```bash
ANALYSIS_TARGET_FPS=15 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

With Docker Compose (root `docker-compose.yml`), this value is passed into the `api` container via:

```yaml
ANALYSIS_TARGET_FPS=${ANALYSIS_TARGET_FPS:-15}
```

## Evaluation and tests

The evaluation helpers live here:

- `app/evaluation/iou.py`
- `app/evaluation/detection_metrics.py`
- `app/evaluation/segmentation_metrics.py`

To run the full test suite:

```bash
cd api
python -m unittest discover -s tests -v
```

The tests cover:

- IoU math (`xyxy` and `xywh`)
- precision / recall / F1 at IoU thresholds
- class-aware and class-agnostic matching
- per-class metrics and confusion matrix
- `mAP@0.5` and `mAP@0.5:0.95`
- segmentation metrics (`Dice`, pixel accuracy, mask IoU, multiclass mIoU)
- API integration and common failure paths

If you want to run the benchmark/evaluation scripts, install the extra deps:

```bash
cd api
pip install -r requirements-dev.txt
```

### Detection report (IoU/F1/mAP/per-class/confusion)

Sample input: `examples/detection_eval_sample.json`

```bash
cd api
python scripts/run_detection_eval.py --input examples/detection_eval_sample.json
```

Save report:

```bash
python scripts/run_detection_eval.py --input examples/detection_eval_sample.json --output reports/detection_metrics.json
```

### Segmentation report (Dice / Pixel Accuracy / mIoU)

Sample input: `examples/segmentation_eval_sample.json`

```bash
cd api
python scripts/run_segmentation_eval.py --input examples/segmentation_eval_sample.json
```

Save report:

```bash
python scripts/run_segmentation_eval.py --input examples/segmentation_eval_sample.json --output reports/segmentation_metrics.json
```

### Latency and FPS benchmark

```bash
cd api
python scripts/run_latency_benchmark.py --model ../yolo11n.pt --source 0 --frames 120 --warmup 20
```

`--source` examples:

- webcam: `0`
- video file: `../some_clip.mp4`
- backend snapshot URL: `http://localhost:8000/debug/sessions/<session_id>/latest.jpg`

### Robustness evaluation (blur / brightness / contrast / jpeg artifacts / noise)

Input JSON format:

```json
{
  "images": [
    {
      "image_id": "img-001",
      "path": "C:/path/to/image.jpg",
      "ground_truths": [
        { "label": "person", "bbox": [10, 20, 130, 260] }
      ]
    }
  ]
}
```

Run:

```bash
cd api
python scripts/run_robustness_eval.py --model ../yolo11n.pt --input C:/path/to/robustness_dataset.json --output reports/robustness_report.json
```

### Regression check

Compare current metrics with a baseline:

```bash
cd api
python scripts/run_regression_check.py --baseline reports/baseline_detection_metrics.json --current reports/current_detection_metrics.json --max-map50-drop 0.02 --max-map5095-drop 0.02
```

Set minimum required thresholds:

```bash
python scripts/run_regression_check.py --current reports/current_detection_metrics.json --min-map50 0.40 --min-map5095 0.20
```

### API integration and failure-path tests

`tests/test_api_integration.py` checks:

- `/health`
- `/debug/sessions`
- `/debug/sessions/history`
- unknown session snapshot returns `404`
- no-snapshot-yet returns `404`
- unknown ICE session returns `404`
- malformed offer payload returns `422`

## API contract

### `POST /webrtc/offer`

Request:

```json
{
  "sdp": "...",
  "type": "offer",
  "session_id": "optional"
}
```

Response:

```json
{
  "sdp": "...",
  "type": "answer",
  "session_id": "uuid"
}
```

### `POST /webrtc/ice`

Request:

```json
{
  "session_id": "uuid",
  "candidate": "candidate:...",
  "sdpMid": "0",
  "sdpMLineIndex": 0
}
```

Response:

```json
{
  "ok": true
}
```

### `GET /debug/sessions`

Returns per-session diagnostics, including:

- `analysis_target_fps`
- `incoming_fps`
- `processed_fps`
- `total_frames`
- `processed_frames`
- `dropped_frames`
- snapshot metadata and connection state

### `GET /debug/sessions/{session_id}/latest.jpg`

Returns latest JPEG created from incoming video track.

### `GET /debug/sessions/history`

Returns persisted session artifact metadata (`session.json`) for recent sessions.

## How frame processing works today

In `api/app/main.py`, each video frame is received in `consume_frames()`.

- Incoming rate is tracked as `incoming_fps`.
- Processing is throttled by `analysis_target_fps` using monotonic timing.
- Frames skipped by throttling increment `dropped_frames`.
- Processed frames increment `processed_frames`.
- Each processed frame is saved as a JPEG dump in that session's artifact directory.
- A snapshot JPEG is periodically saved for debug preview.

Every session creates:

- `frame-*.jpg` files for each processed frame
- `frame-*.json` metadata for each processed frame
- `session.json` metadata with counters, timestamps, dump status, and latest detection state

`frame-*.json` includes:

- `frame_id` / `frame_index`
- `frame_at`
- `directional` context (if available)
- `detections`:
  - `objects` (label, confidence, bbox)
  - `metrics` (`num_detections`, `avg_confidence`, `max_confidence`, `min_confidence`)
  - inference timestamp and staleness (`age_ms`, `is_stale`)

`app/session_artifacts/` is ignored by git and excluded from Docker build context.

To clear artifacts:

```bash
../scripts/clean-session-artifacts.sh
```

This gives you backpressure control before a real model is attached.

## Linking this to a real model

The simplest integration is to replace `send_mock_results()` with model-backed inference results.

Recommended approach:

1. Keep frame sampling where it is (already rate-limited by `ANALYSIS_TARGET_FPS`).
2. Convert sampled frame to model input (`numpy`, PIL, tensor, etc.).
3. Run inference off the event loop (thread pool or worker process for heavy models).
4. Map predictions to the frontend message shape.
5. Send serialized JSON over `session.data_channel`.

Expected message shape (consumed by web client):

```json
{
  "timestamp": "2026-04-02T18:29:21.234Z",
  "guidance_text": "Caution: person ahead.",
  "scene_summary": "Detected one person near the center.",
  "objects": [
    {
      "label": "person",
      "confidence": 0.91,
      "bbox": [0.31, 0.22, 0.22, 0.44]
    }
  ]
}
```

`bbox` is normalized `[x, y, width, height]` in `0..1` coordinates.

## Minimal integration sketch

Below is a high-level pattern (not drop-in complete code):

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(None, model.predict, model_input)

payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "guidance_text": build_guidance(result),
    "scene_summary": summarize_scene(result),
    "objects": to_objects(result),
}

if session.data_channel and getattr(session.data_channel, "readyState", "") == "open":
    session.data_channel.send(json.dumps(payload))
```

## Production notes

- Keep model load as a singleton per process to avoid repeated warmup.
- Avoid blocking `track.recv()` with expensive inference on the main event loop.
- If inference is slower than configured FPS, keep dropping frames rather than queueing unbounded work.
- Consider moving model execution to a separate worker service once load grows.
