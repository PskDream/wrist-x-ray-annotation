# Wrist X-ray Annotation & Tilt Measurement

A tool for annotating lateral wrist X-ray images to measure **Dorsal/Volar Tilt**, with the goal of training a YOLO-pose keypoint detection model.

## Clinical Background

**Dorsal/Volar Tilt** is the inclination angle of the distal radius articular surface.

| Condition | Tilt |
|-----------|------|
| Normal | Volar ~10–15° |
| Colles' fracture | Dorsal (abnormal) |

Measured from **4 keypoints** on a lateral wrist X-ray:

| # | Keypoint | Description |
|---|----------|-------------|
| 1 | `radius_axis_proximal` | Upper point on radius shaft (toward elbow) |
| 2 | `radius_axis_distal` | Lower point on radius shaft (toward wrist) |
| 3 | `volar_lip` | Palmar edge of the articular surface |
| 4 | `dorsal_lip` | Dorsal edge of the articular surface |

## Dataset

```
images/
├── fracture/       ~122 images
└── non-fracture/   ~109 images
                    ~231 images total (mix of left/right wrists)
```

> Left/right wrist images are mirror images — use horizontal flip augmentation during training rather than training separate models.

## Project Structure

```
project/
├── annotate.py       # Annotation tool (OpenCV)
├── images/           # Original X-ray images
├── annotations/      # JSON output (auto-generated)
└── labels/           # YOLO-pose .txt output (auto-generated)
```

## Setup

```bash
pip install opencv-python numpy
```

## Usage

```bash
python annotate.py
```

### Controls

| Key | Action |
|-----|--------|
| Click | Place keypoint (sequential: 1 → 4) |
| L | Set side = Left |
| R | Set side = Right |
| S | Save annotation + compute tilt angle |
| N | Next image |
| B | Back to previous image |
| C | Clear (reset all points) |
| Q / ESC | Quit |

### Keypoint Click Order

Always click in this order:

1. `radius_axis_proximal` — red
2. `radius_axis_distal` — blue
3. `volar_lip` — green
4. `dorsal_lip` — yellow

> Set the side (L/R) before saving each image. Previously annotated images auto-load their saved keypoints when reopened.

## Output Format

### JSON — `annotations/<stem>.json`

```json
{
  "image_id": "wrist_001",
  "image_path": "./images/wrist_001.jpg",
  "side": "right",
  "keypoints": {
    "radius_axis_proximal": [x, y],
    "radius_axis_distal":   [x, y],
    "volar_lip":            [x, y],
    "dorsal_lip":           [x, y]
  },
  "tilt_angle": 15.3,
  "tilt_direction": "volar",
  "annotated_at": "2026-04-19T10:00:00"
}
```

### YOLO-pose — `labels/<stem>.txt`

```
# class cx cy bw bh  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
0 0.5 0.5 1.0 1.0 0.45 0.3 2 0.46 0.7 2 0.38 0.75 2 0.52 0.72 2
```

- `class = 0` (wrist)
- `bbox` = full image (`cx=0.5, cy=0.5, w=1.0, h=1.0`)
- `visibility = 2` (visible)

## ML Roadmap

### Phase 1 — Manual Annotation (current)
Annotate all ~231 images using `annotate.py`.

### Phase 2 — Train YOLO-pose
- Model: YOLOv8-pose (Ultralytics)
- Augmentation: HorizontalFlip, BrightnessContrast, GaussianNoise
- Split: 80% train / 20% val

### Phase 3 — Inference UI
- Load model → predict 4 keypoints → compute tilt automatically
- Physician can adjust keypoints manually if needed