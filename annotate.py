"""
Wrist X-ray Annotation Tool
============================
Controls:
  Click        : Place keypoint (in order: axis_proximal → axis_distal → volar_lip → dorsal_lip)
  L / R        : Set side (Left / Right) — do this before saving
  S            : Save annotation + auto calculate tilt angle
  N            : Next image
  B            : Back to previous image
  R            : Reset current annotation
  Q / ESC      : Quit

Keypoints (click in this order):
  1. radius_axis_proximal  (RED)
  2. radius_axis_distal    (BLUE)
  3. volar_lip             (GREEN)
  4. dorsal_lip            (YELLOW)
"""

import cv2
import numpy as np
import json
import os
import math
import sys
from datetime import datetime
from glob import glob

# ─── Config ───────────────────────────────────────────────────────────────────
IMAGE_DIR   = "./images"          # <-- put your images here
OUTPUT_DIR  = "./annotations"
YOLO_DIR    = "./labels"
IMG_EXTS    = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

KEYPOINT_NAMES = [
    "radius_axis_proximal",
    "radius_axis_distal",
    "volar_lip",
    "dorsal_lip",
]
COLORS = [
    (0, 0, 255),    # RED   - axis_proximal
    (255, 0, 0),    # BLUE  - axis_distal
    (0, 255, 0),    # GREEN - volar_lip
    (0, 255, 255),  # YELLOW- dorsal_lip
]
POINT_RADIUS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_images(folder):
    files = []
    for ext in IMG_EXTS:
        files += glob(os.path.join(folder, ext))
    return sorted(files)


def load_annotation(img_path):
    """Load existing annotation if any."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(OUTPUT_DIR, stem + ".json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            return json.load(f)
    return None


def compute_tilt(points):
    """
    points: dict with keys matching KEYPOINT_NAMES
    Returns (angle_deg, direction) or (None, None)
    """
    try:
        A = np.array(points["radius_axis_proximal"], dtype=float)
        B = np.array(points["radius_axis_distal"],   dtype=float)
        C = np.array(points["volar_lip"],             dtype=float)
        D = np.array(points["dorsal_lip"],            dtype=float)

        # Axis vector
        axis = B - A
        axis_len = np.linalg.norm(axis)
        if axis_len == 0:
            return None, None

        # Perpendicular to axis (white line)
        perp = np.array([-axis[1], axis[0]]) / axis_len

        # Tilt line vector (volar_lip → dorsal_lip)
        tilt_vec = D - C
        tilt_len = np.linalg.norm(tilt_vec)
        if tilt_len == 0:
            return None, None
        tilt_unit = tilt_vec / tilt_len

        # Angle between perp and tilt line
        dot = np.clip(np.dot(perp, tilt_unit), -1.0, 1.0)
        angle = math.degrees(math.acos(abs(dot)))

        # Direction: check which side volar_lip is relative to axis
        # Cross product of axis with (volar_lip - axis_distal)
        v = C - B
        cross = axis[0] * v[1] - axis[1] * v[0]
        direction = "volar" if cross > 0 else "dorsal"

        return round(angle, 1), direction
    except Exception:
        return None, None


def save_annotation(img_path, points, side, img_shape):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(YOLO_DIR, exist_ok=True)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    h, w = img_shape[:2]

    angle, direction = compute_tilt(points)

    # ── JSON ──────────────────────────────────────────────────────────────────
    ann = {
        "image_id":      stem,
        "image_path":    img_path,
        "side":          side,
        "keypoints":     points,
        "tilt_angle":    angle,
        "tilt_direction": direction,
        "annotated_at":  datetime.now().isoformat(),
    }
    json_path = os.path.join(OUTPUT_DIR, stem + ".json")
    with open(json_path, "w") as f:
        json.dump(ann, f, indent=2)

    # ── YOLO-pose .txt ────────────────────────────────────────────────────────
    # Format: class cx cy bw bh  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
    # We use class 0, bbox = full image, visibility = 2 (visible)
    kp_flat = []
    for name in KEYPOINT_NAMES:
        px, py = points[name]
        kp_flat += [px / w, py / h, 2]  # normalised + visible

    line = "0 0.5 0.5 1.0 1.0 " + " ".join(f"{v:.6f}" for v in kp_flat)
    txt_path = os.path.join(YOLO_DIR, stem + ".txt")
    with open(txt_path, "w") as f:
        f.write(line + "\n")

    return angle, direction


def draw_overlay(canvas, points, side, img_path):
    """Draw keypoints, lines, angle info on canvas."""
    h, w = canvas.shape[:2]
    idx = os.path.basename(img_path)

    # ── Draw axis line (black) ─────────────────────────────────────────────
    if "radius_axis_proximal" in points and "radius_axis_distal" in points:
        A = tuple(map(int, points["radius_axis_proximal"]))
        B = tuple(map(int, points["radius_axis_distal"]))
        cv2.line(canvas, A, B, (30, 30, 30), 2)

        # Perpendicular (white line) through midpoint
        mid = ((A[0]+B[0])//2, (A[1]+B[1])//2)
        dx = B[0] - A[0]; dy = B[1] - A[1]
        length = math.hypot(dx, dy) or 1
        px, py = -dy/length, dx/length
        ext = 100
        p1 = (int(mid[0] + px*ext), int(mid[1] + py*ext))
        p2 = (int(mid[0] - px*ext), int(mid[1] - py*ext))
        cv2.line(canvas, p1, p2, (255, 255, 255), 2)

    # ── Draw tilt line (yellow) ────────────────────────────────────────────
    if "volar_lip" in points and "dorsal_lip" in points:
        C = tuple(map(int, points["volar_lip"]))
        D = tuple(map(int, points["dorsal_lip"]))
        cv2.line(canvas, C, D, (0, 215, 255), 2)

    # ── Draw each keypoint ─────────────────────────────────────────────────
    for i, name in enumerate(KEYPOINT_NAMES):
        if name not in points:
            break
        pt = tuple(map(int, points[name]))
        cv2.circle(canvas, pt, POINT_RADIUS, COLORS[i], -1)
        cv2.circle(canvas, pt, POINT_RADIUS+2, (255,255,255), 1)
        label = f"{i+1}.{name.replace('radius_','').replace('_',' ')}"
        cv2.putText(canvas, label, (pt[0]+12, pt[1]+5),
                    FONT, 0.45, COLORS[i], 1, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────
    next_idx = len(points)
    if next_idx < len(KEYPOINT_NAMES):
        next_name = KEYPOINT_NAMES[next_idx]
        hint = f"Click pt {next_idx+1}: {next_name}"
        cv2.putText(canvas, hint, (10, h-15),
                    FONT, 0.55, COLORS[next_idx], 2, cv2.LINE_AA)

    # Tilt angle
    if len(points) == 4:
        angle, direction = compute_tilt(points)
        if angle is not None:
            tilt_txt = f"Tilt: {angle} deg  ({direction})"
            cv2.putText(canvas, tilt_txt, (10, h-45),
                        FONT, 0.65, (0,255,255), 2, cv2.LINE_AA)

    # Side + filename
    side_color = (0,200,255) if side == "right" else (200,100,255)
    cv2.putText(canvas, f"Side: {side.upper()}  [L/R to change]",
                (10, 30), FONT, 0.6, side_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, idx, (10, 55), FONT, 0.5, (180,180,180), 1, cv2.LINE_AA)

    # Legend top-right
    legend = ["S=Save","N=Next","B=Back","C=Clear","L/R=Side","Q=Quit"]
    for li, txt in enumerate(legend):
        cv2.putText(canvas, txt, (w-130, 25+li*22),
                    FONT, 0.48, (200,200,200), 1, cv2.LINE_AA)

    # Saved indicator
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if os.path.exists(os.path.join(OUTPUT_DIR, stem+".json")):
        cv2.putText(canvas, "SAVED", (w-80, h-15),
                    FONT, 0.7, (0,255,100), 2, cv2.LINE_AA)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(IMAGE_DIR):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        print(f"[INFO] Created '{IMAGE_DIR}/' — put your X-ray images there and re-run.")
        sys.exit(0)

    images = get_images(IMAGE_DIR)
    if not images:
        print(f"[ERROR] No images found in '{IMAGE_DIR}/'")
        sys.exit(1)

    print(f"[INFO] Found {len(images)} images.")

    idx      = 0
    points   = {}   # name → [x, y]
    side     = "right"
    msg      = ""
    msg_timer = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            next_pt = len(points)
            if next_pt < len(KEYPOINT_NAMES):
                points[KEYPOINT_NAMES[next_pt]] = [x, y]

    win = "Wrist Annotator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 700)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        img_path = images[idx]
        raw = cv2.imread(img_path)
        if raw is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            idx = (idx + 1) % len(images)
            continue

        # Auto-load existing annotation
        existing = load_annotation(img_path)
        if existing and not points:
            points = existing.get("keypoints", {})
            side   = existing.get("side", "right")

        canvas = raw.copy()
        draw_overlay(canvas, points, side, img_path)

        # Progress
        prog = f"{idx+1}/{len(images)}"
        cv2.putText(canvas, prog, (10, canvas.shape[0]-75),
                    FONT, 0.55, (180,180,180), 1, cv2.LINE_AA)

        # Flash message
        if msg_timer > 0:
            cv2.putText(canvas, msg, (canvas.shape[1]//2-150, 90),
                        FONT, 0.9, (0,255,180), 2, cv2.LINE_AA)
            msg_timer -= 1

        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):   # Q / ESC
            break

        elif key == ord('c'):       # Reset
            points = {}
            msg = "Clear!"; msg_timer = 20

        elif key == ord('l'):       # Side left
            side = "left"

        elif key == ord('r') and False:
            pass

        elif key == ord('R') or (key == ord('r')):
            # distinguish r (reset) already handled above — cv2 is lowercase only
            pass

        # re-check for side keys (lowercase only from cv2)
        if key == ord('l'):
            side = "left"
        elif key == ord('r'):       # 'i' as right (avoid clash)
            side = "right"          # user can press R for right side too via menu

        if key == ord('s'):         # Save
            if len(points) == 4:
                angle, direction = save_annotation(img_path, points, side, raw.shape)
                msg = f"Saved! Tilt={angle}deg ({direction})"
                msg_timer = 40
            else:
                msg = f"Need 4 points! ({len(points)}/4 placed)"
                msg_timer = 30

        elif key == ord('n'):       # Next
            idx = (idx + 1) % len(images)
            points = {}
            existing = load_annotation(images[idx])
            if existing:
                points = existing.get("keypoints", {})
                side   = existing.get("side", side)

        elif key == ord('b'):       # Back
            idx = (idx - 1) % len(images)
            points = {}
            existing = load_annotation(images[idx])
            if existing:
                points = existing.get("keypoints", {})
                side   = existing.get("side", side)

    cv2.destroyAllWindows()
    print("[INFO] Done. Annotations saved to:", OUTPUT_DIR)
    print("[INFO] YOLO labels saved to:", YOLO_DIR)


if __name__ == "__main__":
    main()
