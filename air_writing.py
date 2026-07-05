"""
Virtual Air Writing System
---------------------------
Draw in the air using hand gestures, captured via webcam.

Tech stack: Python, OpenCV, MediaPipe

Features:
  - Real-time hand tracking
  - Finger landmark detection
  - Gesture recognition (draw / select / idle)
  - Virtual air writing on a persistent canvas
  - Multiple brush colors
  - Eraser tool
  - Clear canvas
  - Save drawing to disk
  - Interactive on-screen toolbar

Controls (keyboard, as backup to gestures):
  s = save drawing
  c = clear canvas
  q / ESC = quit
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAM_WIDTH, CAM_HEIGHT = 1280, 720
TOOLBAR_HEIGHT = 90
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40
SAVE_DIR = "saved_drawings"

# Toolbar swatches: (label, BGR color). "Eraser" and "Clear" are special.
PALETTE = [
    ("Magenta", (255, 0, 255)),
    ("Blue",    (255, 0, 0)),
    ("Green",   (0, 255, 0)),
    ("Red",     (0, 0, 255)),
    ("Yellow",  (0, 255, 255)),
    ("Orange",  (0, 165, 255)),
]

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe hands setup
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Landmark indices we care about
TIP_IDS = [4, 8, 12, 16, 20]        # thumb, index, middle, ring, pinky tips
PIP_IDS = [3, 6, 10, 14, 18]        # joint just below each tip (for up/down check)


def fingers_up(landmarks, handedness_label):
    """
    Return a list of 5 booleans indicating whether each finger
    (thumb, index, middle, ring, pinky) is extended ("up").
    """
    fingers = []

    # Thumb: compare x instead of y since it moves sideways, not up/down.
    # Mirror logic depending on left/right hand (image is already flipped
    # so "Right" hand appears on the right side of the mirrored frame).
    if handedness_label == "Right":
        fingers.append(landmarks[4].x > landmarks[3].x)
    else:
        fingers.append(landmarks[4].x < landmarks[3].x)

    # Other four fingers: tip above pip joint (smaller y) = extended
    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return fingers


def draw_toolbar(frame, active_color, mode):
    """Draw the color palette + eraser + clear buttons at the top of the frame."""
    h, w, _ = frame.shape
    swatch_w = w // (len(PALETTE) + 2)  # +2 for Eraser and Clear buttons

    cv2.rectangle(frame, (0, 0), (w, TOOLBAR_HEIGHT), (30, 30, 30), -1)

    boxes = []
    for i, (name, color) in enumerate(PALETTE):
        x1, x2 = i * swatch_w, (i + 1) * swatch_w
        cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), color, -1)
        if color == active_color and mode == "color":
            cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), (255, 255, 255), 4)
        boxes.append((x1, x2, "color", color))

    # Eraser button
    i = len(PALETTE)
    x1, x2 = i * swatch_w, (i + 1) * swatch_w
    cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), (200, 200, 200), -1)
    cv2.putText(frame, "ERASER", (x1 + 5, TOOLBAR_HEIGHT // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    if mode == "eraser":
        cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), (0, 0, 0), 4)
    boxes.append((x1, x2, "eraser", None))

    # Clear button
    i = len(PALETTE) + 1
    x1, x2 = i * swatch_w, (i + 1) * swatch_w
    cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), (60, 60, 220), -1)
    cv2.putText(frame, "CLEAR", (x1 + 15, TOOLBAR_HEIGHT // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    boxes.append((x1, x2, "clear", None))

    return boxes


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    canvas = None
    active_color = PALETTE[0][1]
    mode = "color"          # "color" or "eraser"
    prev_x, prev_y = None, None
    last_save_msg_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        toolbar_boxes = draw_toolbar(frame, active_color, mode)

        drawing_this_frame = False

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            handedness_label = result.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            index_x, index_y = int(lm[8].x * w), int(lm[8].y * h)

            up = fingers_up(lm, handedness_label)
            # up = [thumb, index, middle, ring, pinky]

            index_up = up[1]
            middle_up = up[2]
            other_fingers_down = not up[2] and not up[3] and not up[4]

            # --- Selection gesture: index + middle both up -> hover/select mode ---
            if index_up and middle_up:
                prev_x, prev_y = None, None
                cv2.circle(frame, (index_x, index_y), 12, (255, 255, 255), 2)

                if index_y < TOOLBAR_HEIGHT:
                    for x1, x2, kind, color in toolbar_boxes:
                        if x1 <= index_x <= x2:
                            if kind == "color":
                                active_color = color
                                mode = "color"
                            elif kind == "eraser":
                                mode = "eraser"
                            elif kind == "clear":
                                canvas[:] = 0

            # --- Drawing gesture: only index finger up -> draw/erase ---
            elif index_up and other_fingers_down:
                if index_y > TOOLBAR_HEIGHT:  # don't draw over the toolbar
                    drawing_this_frame = True
                    cv2.circle(frame, (index_x, index_y), 8, active_color, -1)

                    if prev_x is None:
                        prev_x, prev_y = index_x, index_y

                    if mode == "eraser":
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y),
                                  (0, 0, 0), ERASER_THICKNESS)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y),
                                  active_color, BRUSH_THICKNESS)

                    prev_x, prev_y = index_x, index_y
                else:
                    prev_x, prev_y = None, None
            else:
                # Any other gesture (fist, open palm, etc.) = pen up
                prev_x, prev_y = None, None

        else:
            prev_x, prev_y = None, None

        # Merge canvas onto live frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
        combined = cv2.add(frame_bg, canvas)

        # Small "saved!" confirmation text
        if time.time() - last_save_msg_time < 1.2:
            cv2.putText(combined, "Saved!", (w - 160, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Virtual Air Writing System", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:      # q or ESC
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
            cv2.imwrite(filename, canvas)
            last_save_msg_time = time.time()
            print(f"Saved drawing to {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
