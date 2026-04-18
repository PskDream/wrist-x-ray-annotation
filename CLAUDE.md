# Wrist X-ray Annotation & Tilt Measurement Project

## Project Overview

A tool for annotating lateral wrist X-ray images to measure Dorsal/Volar Tilt.
The end goal is to train a YOLO-pose keypoint detection model.

## Clinical Background

**Dorsal/Volar Tilt** is the inclination angle of the distal radius articular surface.
- Normal = volar tilt ~10–15 degrees
- Colles' fracture = dorsal tilt (abnormal)

Measured from 4 keypoints on a lateral wrist X-ray:
1. `radius_axis_proximal` — upper point on radius shaft (toward elbow)
2. `radius_axis_distal` — lower point on radius shaft (toward wrist)
3. `volar_lip` — palmar edge of the articular surface
4. `dorsal_lip` — dorsal edge of the articular surface

## Dataset

```
images/          ← wrist X-ray images (JPEG/PNG)
├── fracture     ~122 images
└── non-fracture ~109 images
total ~231 images (mix of left and right wrists)
```

**Note:** Left/right wrist images are mirror images of each other →
use horizontal flip augmentation during training instead of training separate models.

## Directory Structure

```
project/
├── CLAUDE.md
├── annotate.py          ← annotation tool (OpenCV)
├── images/              ← original X-ray images
├── annotations/         ← JSON output (auto-generated)
└── labels/              ← YOLO-pose .txt output (auto-generated)
```

## Annotation Tool (`annotate.py`)

### Dependencies
```bash
pip install opencv-python numpy
```

### Run
```bash
python annotate.py
```

### Controls
| Key | Action |
|-----|--------|
| Click | Place keypoint (sequential, 1→4 automatically) |
| L | Set side = Left |
| R | Set side = Right |
| S | Save annotation + compute tilt angle |
| N | Next image |
| B | Back to previous image |
| C | Clear (reset all points) |
| Q / ESC | Quit |

### Keypoint Click Order
Always click in this order:
1. 🔴 `radius_axis_proximal`
2. 🔵 `radius_axis_distal`
3. 🟢 `volar_lip`
4. 🟡 `dorsal_lip`

## Output Format

### JSON (`annotations/<stem>.json`)
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

### YOLO-pose (`labels/<stem>.txt`)
```
# class cx cy bw bh  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
0 0.5 0.5 1.0 1.0 0.45 0.3 2 0.46 0.7 2 0.38 0.75 2 0.52 0.72 2
```
- class = 0 (wrist)
- bbox = full image (cx=0.5, cy=0.5, w=1.0, h=1.0)
- visibility = 2 (visible)

## ML Plan

### Phase 1 (current) — Manual Annotation
- Annotate all images using `annotate.py`
- Target: complete all 231 images

### Phase 2 — Train YOLO-pose
- Model: YOLO26-pose (Ultralytics)
- Augmentation: HorizontalFlip, BrightnessContrast, GaussianNoise
- Split: 80% train / 20% val

### Phase 3 — Inference UI
- Load model → predict 4 keypoints → compute tilt automatically
- Physician can adjust keypoints manually if model prediction is off

## Conventions

- Keypoint names must not be changed (must match YOLO-pose schema)
- Side (L/R) must be set before saving each image (default = right)
- Previously annotated images auto-load their saved keypoints when reopened