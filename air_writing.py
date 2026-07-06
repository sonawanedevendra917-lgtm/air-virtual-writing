"""
Virtual Air Writing System - v3
---------------------------------
Draw in the air using hand gestures, captured via webcam.

New in v3 (on top of v2's fullscreen / undo / brush sizes / FPS):

  EASY WINS
  - Pinch-to-resize brush: pinch thumb + index (other fingers down) and move
    them apart/together to shrink/grow the brush live.
  - Live color/size preview circle at the fingertip.
  - '?' shows an on-screen keyboard shortcuts overlay.
  - Dwell-to-confirm: hovering a button now has to be held briefly (with a
    filling progress ring) before it triggers - prevents accidental clears.

  MEDIUM
  - Two-hand support: one hand can draw while the other operates the toolbar,
    at the same time.
  - Shape snapping: press 'v' to toggle "vector mode" - your next completed
    stroke gets snapped to a clean line / rectangle / circle if it's close
    enough to one.
  - Multi-page canvas: 'n' next page, 'b' previous page. Like a whiteboard
    with multiple sheets.
  - OCR text mode: press 't' to run OCR on the current canvas and print the
    recognized text on screen (requires pytesseract + Tesseract installed;
    the app still works fine without it, this feature just no-ops).
  - Session recording: press 'r' to start/stop recording; stopping saves an
    animated GIF of your session to saved_drawings/.

Tech stack: Python, OpenCV, MediaPipe (+ optional: pytesseract, imageio)

Controls:
  f = fullscreen        h = toggle hand skeleton      ? = shortcuts overlay
  u = undo              c = clear current page        v = toggle vector/snap mode
  s = save PNG          n = next page                  b = previous page
  t = OCR current page  r = start/stop GIF recording   q / Esc = quit
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math

try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAM_WIDTH, CAM_HEIGHT = 1280, 720
TITLE_BAR_HEIGHT = 50
TOOLBAR_HEIGHT = 80
TOP_UI_HEIGHT = TITLE_BAR_HEIGHT + TOOLBAR_HEIGHT
SAVE_DIR = "saved_drawings"
MAX_UNDO_STATES = 20
NUM_PAGES = 3

MIN_BRUSH, MAX_BRUSH = 3, 40
BRUSH_SIZES = {"S": 4, "M": 9, "L": 16}
ERASER_MULTIPLIER = 5

DWELL_TIME = 0.7          # seconds to hover a button before it triggers
PINCH_ON_DIST = 45        # pixels: thumb-index distance below this = "pinched"
RECORD_EVERY_N_FRAMES = 3  # only keep every Nth frame for the GIF

PALETTE = [
    ("Magenta", (255, 0, 255)),
    ("Blue",    (255, 0, 0)),
    ("Green",   (0, 255, 0)),
    ("Red",     (0, 0, 255)),
    ("Yellow",  (0, 255, 255)),
    ("Orange",  (0, 165, 255)),
    ("White",   (255, 255, 255)),
]

BG_DARK = (24, 24, 24)
BG_PANEL = (38, 38, 38)
ACCENT = (80, 200, 255)

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe hands setup (2 hands now, for two-hand support)
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]


def fingers_up(landmarks, handedness_label):
    fingers = []
    if handedness_label == "Right":
        fingers.append(landmarks[4].x > landmarks[3].x)
    else:
        fingers.append(landmarks[4].x < landmarks[3].x)
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)
    return fingers


def rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)


def point_in_box(px, py, box):
    (x1, y1), (x2, y2), _ = box
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_dwell_ring(img, center, progress, radius=22, color=ACCENT):
    angle = int(360 * progress)
    cv2.ellipse(img, center, (radius, radius), -90, 0, angle, color, 4, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Shape snapping helpers
# ---------------------------------------------------------------------------

def snap_stroke(points):
    """
    Classify a completed stroke as a line, rectangle, or circle.
    Returns (shape_type, data) or None if nothing matches confidently.
    """
    if len(points) < 8:
        return None

    pts = np.array(points, dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bbox_w, bbox_h = x1 - x0, y1 - y0
    diag = math.hypot(bbox_w, bbox_h)
    if diag < 20:
        return None

    p_start, p_end = pts[0], pts[-1]
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)
    if line_len > 1e-3:
        line_dir = line_vec / line_len
        rel = pts - p_start
        proj_len = rel @ line_dir
        proj_pts = p_start + np.outer(proj_len, line_dir)
        dists = np.linalg.norm(pts - proj_pts, axis=1)
        if dists.max() < 0.08 * diag + 6:
            return ("line", (tuple(p_start.astype(int)), tuple(p_end.astype(int))))

    center = pts.mean(axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    mean_r = radii.mean()
    if mean_r > 8 and radii.std() < 0.18 * mean_r:
        return ("circle", (tuple(center.astype(int)), int(mean_r)))

    edge_hits = 0
    for px, py in pts:
        near_edge = (
            abs(px - x0) < 0.1 * bbox_w + 6 or abs(px - x1) < 0.1 * bbox_w + 6 or
            abs(py - y0) < 0.1 * bbox_h + 6 or abs(py - y1) < 0.1 * bbox_h + 6
        )
        if near_edge:
            edge_hits += 1
    if bbox_w > 20 and bbox_h > 20 and edge_hits / len(pts) > 0.7:
        return ("rect", ((int(x0), int(y0)), (int(x1), int(y1))))

    return None


def draw_snapped_shape(canvas, shape, color, thickness):
    kind, data = shape
    if kind == "line":
        cv2.line(canvas, data[0], data[1], color, thickness, cv2.LINE_AA)
    elif kind == "circle":
        cv2.circle(canvas, data[0], data[1], color, thickness, cv2.LINE_AA)
    elif kind == "rect":
        cv2.rectangle(canvas, data[0], data[1], color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# UI drawing
# ---------------------------------------------------------------------------

def draw_ui(frame, state):
    h, w, _ = frame.shape
    boxes = {}

    cv2.rectangle(frame, (0, 0), (w, TITLE_BAR_HEIGHT), BG_DARK, -1)
    cv2.putText(frame, "VIRTUAL AIR WRITING SYSTEM", (16, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    right_bits = [
        f"Mode: {state['mode'].upper()}",
        f"Brush: {state['brush_size']}px",
        f"Page: {state['page'] + 1}/{NUM_PAGES}",
    ]
    if state["vector_mode"]:
        right_bits.append("VECTOR")
    if state["recording"]:
        right_bits.append("REC")
    text = "   ".join(right_bits)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    color = (60, 60, 255) if state["recording"] else ACCENT
    cv2.putText(frame, text, (w - tw - 20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    y0 = TITLE_BAR_HEIGHT
    cv2.rectangle(frame, (0, y0), (w, y0 + TOOLBAR_HEIGHT), BG_PANEL, -1)

    pad = 10
    btn_h = TOOLBAR_HEIGHT - 2 * pad
    x = pad

    swatch_w = 56
    for name, color_val in PALETTE:
        pt1, pt2 = (x, y0 + pad), (x + swatch_w, y0 + pad + btn_h)
        rounded_rect(frame, pt1, pt2, color_val, radius=8)
        if state["mode"] == "color" and state["active_color"] == color_val:
            rounded_rect(frame, pt1, pt2, (255, 255, 255), radius=8, thickness=3)
        boxes[f"color:{name}"] = (pt1, pt2, ("color", color_val))
        x += swatch_w + 6

    x += 10
    pt1, pt2 = (x, y0 + pad), (x + 86, y0 + pad + btn_h)
    rounded_rect(frame, pt1, pt2, (200, 200, 200), radius=8)
    if state["mode"] == "eraser":
        rounded_rect(frame, pt1, pt2, (0, 0, 0), radius=8, thickness=3)
    cv2.putText(frame, "ERASER", (pt1[0] + 6, pt1[1] + btn_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    boxes["eraser"] = (pt1, pt2, ("eraser", None))
    x += 96

    for label, val in BRUSH_SIZES.items():
        pt1, pt2 = (x, y0 + pad), (x + 40, y0 + pad + btn_h)
        rounded_rect(frame, pt1, pt2, (90, 90, 90), radius=8)
        if state["brush_size"] == val:
            rounded_rect(frame, pt1, pt2, ACCENT, radius=8, thickness=3)
        cv2.putText(frame, label, (pt1[0] + 12, pt1[1] + btn_h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        boxes[f"size:{label}"] = (pt1, pt2, ("size", val))
        x += 48

    x += 10
    for label, kind, color_val in [("UNDO", "undo", (210, 160, 40)),
                                    ("CLEAR", "clear", (60, 60, 220)),
                                    ("SAVE", "save", (40, 170, 90)),
                                    ("VEC", "vector", (150, 90, 200))]:
        pt1, pt2 = (x, y0 + pad), (x + 70, y0 + pad + btn_h)
        rounded_rect(frame, pt1, pt2, color_val, radius=8)
        if kind == "vector" and state["vector_mode"]:
            rounded_rect(frame, pt1, pt2, (255, 255, 255), radius=8, thickness=3)
        text_color = (0, 0, 0) if kind == "undo" else (255, 255, 255)
        cv2.putText(frame, label, (pt1[0] + 6, pt1[1] + btn_h // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 2, cv2.LINE_AA)
        boxes[kind] = (pt1, pt2, (kind, None))
        x += 78

    return boxes


HELP_LINES = [
    "f = fullscreen        h = hand skeleton on/off",
    "u = undo              c = clear current page",
    "s = save PNG          v = vector/shape-snap mode",
    "n = next page         b = previous page",
    "t = OCR this page     r = start/stop GIF recording",
    "q / Esc = quit         ? = hide this overlay",
]


def draw_help_overlay(img):
    h, w, _ = img.shape
    box_w, box_h = 560, 40 + 28 * len(HELP_LINES)
    x0, y0 = (w - box_w) // 2, (h - box_h) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), ACCENT, 2)
    cv2.putText(img, "KEYBOARD SHORTCUTS", (x0 + 20, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(HELP_LINES):
        cv2.putText(img, line, (x0 + 20, y0 + 66 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    window_name = "Virtual Air Writing System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False
    show_skeleton = True
    show_help = False

    pages = None
    state = {
        "mode": "color",
        "active_color": PALETTE[0][1],
        "brush_size": BRUSH_SIZES["M"],
        "page": 0,
        "vector_mode": False,
        "recording": False,
    }

    draw_state = {0: {"prev": None, "stroke_pts": [], "active": False},
                  1: {"prev": None, "stroke_pts": [], "active": False}}

    undo_stacks = [[] for _ in range(NUM_PAGES)]

    hover_box = None
    hover_start = 0

    last_msg, last_msg_time = "", 0
    prev_time = time.time()
    fps = 0

    recorded_frames = []
    frame_counter = 0

    def show_message(text):
        nonlocal last_msg, last_msg_time
        last_msg, last_msg_time = text, time.time()

    def push_undo(page_idx, canvas_arr):
        stack = undo_stacks[page_idx]
        stack.append(canvas_arr.copy())
        if len(stack) > MAX_UNDO_STATES:
            stack.pop(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if pages is None:
            pages = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(NUM_PAGES)]

        canvas = pages[state["page"]]

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        boxes = draw_ui(frame, state)
        eraser_thickness = state["brush_size"] * ERASER_MULTIPLIER

        active_hover_key = None
        active_hover_pos = None

        if result.multi_hand_landmarks:
            for slot, hand_landmarks in enumerate(result.multi_hand_landmarks[:2]):
                handedness_label = result.multi_handedness[slot].classification[0].label
                if show_skeleton:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                index_x, index_y = int(lm[8].x * w), int(lm[8].y * h)
                thumb_x, thumb_y = int(lm[4].x * w), int(lm[4].y * h)
                pinch_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)

                up = fingers_up(lm, handedness_label)
                index_up, middle_up = up[1], up[2]
                other_fingers_down = not up[2] and not up[3] and not up[4]
                d = draw_state[slot]

                if other_fingers_down and pinch_dist < PINCH_ON_DIST + 25 and not index_up:
                    new_size = int(np.interp(pinch_dist, [10, 120], [MIN_BRUSH, MAX_BRUSH]))
                    state["brush_size"] = max(MIN_BRUSH, min(MAX_BRUSH, new_size))
                    mid_x, mid_y = (index_x + thumb_x) // 2, (index_y + thumb_y) // 2
                    cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), ACCENT, 2)
                    cv2.circle(frame, (mid_x, mid_y), state["brush_size"] // 2 + 2, state["active_color"], -1)
                    d["prev"] = None
                    d["active"] = False
                    continue

                if index_up and middle_up:
                    d["prev"] = None
                    d["active"] = False
                    cv2.circle(frame, (index_x, index_y), 10, (255, 255, 255), 2)

                    if index_y < TOP_UI_HEIGHT:
                        for key, box in boxes.items():
                            if point_in_box(index_x, index_y, box):
                                active_hover_key = key
                                active_hover_pos = (index_x, index_y)
                                break

                elif index_up and other_fingers_down:
                    if index_y > TOP_UI_HEIGHT:
                        if not d["active"]:
                            push_undo(state["page"], canvas)
                            d["active"] = True
                            d["stroke_pts"] = []

                        color = (0, 0, 0) if state["mode"] == "eraser" else state["active_color"]
                        thickness = eraser_thickness if state["mode"] == "eraser" else state["brush_size"]

                        cv2.circle(frame, (index_x, index_y), thickness // 2 + 2, state["active_color"], -1)

                        if d["prev"] is None:
                            d["prev"] = (index_x, index_y)
                        cv2.line(canvas, d["prev"], (index_x, index_y), color, thickness, cv2.LINE_AA)
                        d["stroke_pts"].append((index_x, index_y))
                        d["prev"] = (index_x, index_y)
                    else:
                        d["prev"] = None
                        d["active"] = False
                else:
                    if d["active"] and state["vector_mode"] and state["mode"] != "eraser":
                        shape = snap_stroke(d["stroke_pts"])
                        if shape:
                            draw_snapped_shape(canvas, shape, state["active_color"], state["brush_size"])
                            show_message(f"Snapped to {shape[0]}")
                    d["prev"] = None
                    d["active"] = False
                    d["stroke_pts"] = []
        else:
            for d in draw_state.values():
                d["prev"] = None
                d["active"] = False

        if active_hover_key is not None:
            if hover_box == active_hover_key:
                elapsed = time.time() - hover_start
            else:
                hover_box, hover_start = active_hover_key, time.time()
                elapsed = 0

            progress = min(elapsed / DWELL_TIME, 1.0)
            draw_dwell_ring(frame, active_hover_pos, progress)

            if progress >= 1.0:
                _, _, (kind, value) = boxes[active_hover_key]
                if kind == "color":
                    state["mode"], state["active_color"] = "color", value
                elif kind == "eraser":
                    state["mode"] = "eraser"
                elif kind == "size":
                    state["brush_size"] = value
                elif kind == "undo":
                    stack = undo_stacks[state["page"]]
                    if stack:
                        pages[state["page"]][:] = stack.pop()
                        show_message("Undone")
                elif kind == "clear":
                    push_undo(state["page"], canvas)
                    canvas[:] = 0
                    show_message("Page cleared")
                elif kind == "save":
                    filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
                    cv2.imwrite(filename, canvas)
                    show_message("Saved!")
                elif kind == "vector":
                    state["vector_mode"] = not state["vector_mode"]
                    show_message("Vector mode " + ("ON" if state["vector_mode"] else "OFF"))
                hover_box, hover_start = None, 0
        else:
            hover_box, hover_start = None, 0

        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
        combined = cv2.add(frame_bg, canvas)
        combined[:TOP_UI_HEIGHT, :] = frame[:TOP_UI_HEIGHT, :]

        cv2.putText(combined, f"FPS: {int(fps)}", (w - 110, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        if state["recording"]:
            cv2.circle(combined, (30, h - 24), 8, (0, 0, 255), -1)
            cv2.putText(combined, "REC", (46, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if time.time() - last_msg_time < 1.4 and last_msg:
            (tw, th), _ = cv2.getTextSize(last_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cv2.rectangle(combined, (w // 2 - tw // 2 - 16, h - 70),
                          (w // 2 + tw // 2 + 16, h - 30), (0, 0, 0), -1)
            cv2.putText(combined, last_msg, (w // 2 - tw // 2, h - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

        if show_help:
            draw_help_overlay(combined)

        if state["recording"]:
            frame_counter += 1
            if frame_counter % RECORD_EVERY_N_FRAMES == 0:
                small = cv2.resize(combined, (combined.shape[1] // 2, combined.shape[0] // 2))
                recorded_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            push_undo(state["page"], canvas)
            canvas[:] = 0
            show_message("Page cleared")
        elif key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
            cv2.imwrite(filename, canvas)
            show_message("Saved!")
        elif key == ord('u'):
            stack = undo_stacks[state["page"]]
            if stack:
                pages[state["page"]][:] = stack.pop()
                show_message("Undone")
        elif key == ord('h'):
            show_skeleton = not show_skeleton
        elif key == ord('v'):
            state["vector_mode"] = not state["vector_mode"]
            show_message("Vector mode " + ("ON" if state["vector_mode"] else "OFF"))
        elif key == ord('n'):
            state["page"] = (state["page"] + 1) % NUM_PAGES
            show_message(f"Page {state['page'] + 1}/{NUM_PAGES}")
        elif key == ord('b'):
            state["page"] = (state["page"] - 1) % NUM_PAGES
            show_message(f"Page {state['page'] + 1}/{NUM_PAGES}")
        elif key == ord('?') or key == ord('/'):
            show_help = not show_help
        elif key == ord('t'):
            if HAVE_TESSERACT:
                gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(inv).strip()
                show_message(f"OCR: {text[:40] if text else '(nothing recognized)'}")
            else:
                show_message("Install pytesseract + Tesseract for OCR")
        elif key == ord('r'):
            if not state["recording"]:
                state["recording"] = True
                recorded_frames.clear()
                frame_counter = 0
                show_message("Recording started")
            else:
                state["recording"] = False
                if HAVE_IMAGEIO and recorded_frames:
                    gif_path = os.path.join(SAVE_DIR, f"session_{int(time.time())}.gif")
                    imageio.mimsave(gif_path, recorded_frames, fps=10)
                    show_message(f"Saved GIF: {gif_path}")
                elif not HAVE_IMAGEIO:
                    show_message("Install imageio to export GIFs")
                else:
                    show_message("Recording stopped (no frames captured)")
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, CAM_WIDTH, CAM_HEIGHT)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
