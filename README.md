# Virtual Air Writing System ✍️

A real-time computer vision app that lets you "write" in the air using
hand gestures captured through your webcam — no touch screen or stylus
required.

Built with **Python**, **OpenCV**, and **MediaPipe**.

## Features

- ✅ Real-time hand tracking
- ✅ Finger landmark detection (21 points per hand)
- ✅ Gesture recognition (draw / select / idle)
- ✅ Virtual air writing on a persistent canvas
- ✅ Multiple brush colors
- ✅ Eraser tool
- ✅ Clear canvas option
- ✅ Save drawing feature
- ✅ Interactive on-screen toolbar

## How the gestures work

| Gesture | Meaning |
|---|---|
| ☝️ Only index finger up | **Draw mode** — moves the brush and draws a line |
| ✌️ Index + middle finger up | **Select mode** — hover over the toolbar to pick a color / eraser / clear, without drawing |
| ✊ Fist / other fingers up | **Pen up** — stops drawing (like lifting a pen off paper) |

The toolbar sits at the top of the frame. Point at a color swatch, the
eraser button, or the clear button in select mode to activate it.

Keyboard shortcuts (handy for demos or if gesture detection misfires):
- `s` — save the current canvas to `saved_drawings/`
- `c` — clear the canvas
- `q` or `Esc` — quit

## How it works (for your LinkedIn write-up)

1. **Hand tracking & landmarks** — MediaPipe's Hands solution detects 21
   3D landmarks per hand every frame (fingertips, joints, wrist).
2. **Gesture recognition** — for each frame, the app checks whether each
   fingertip landmark is above or below its corresponding joint (PIP)
   landmark to decide if that finger is "up" or "down." Combinations of
   up/down fingers map to draw / select / idle states.
3. **Virtual canvas** — drawing doesn't happen directly on the webcam
   frame. It happens on a separate black canvas (a NumPy array) that
   persists across frames. Each new frame, the canvas is masked and
   composited on top of the live video feed, so strokes stay visible
   as you move.
4. **Toolbar & tool selection** — the color palette, eraser, and clear
   button are just rectangles drawn each frame with OpenCV. In select
   mode, the app checks whether your fingertip's (x, y) position falls
   inside one of those rectangles.
5. **Persistence** — pressing `s` (or building a save-gesture) writes
   the canvas array to disk as a PNG with OpenCV.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run it
python air_writing.py
```

Requires a working webcam. Tested with Python 3.9–3.11 (MediaPipe does
not yet support every Python version, so if install fails, try 3.10).

## Ideas to extend it (great for a v2 LinkedIn post)

- Add a "pinch to change brush size" gesture (thumb + index distance)
- Recognize simple shapes (circle, line, square) and auto-clean them
- Add two-hand support (one hand draws, other hand controls toolbar)
- Export drawing as a short GIF of the drawing process, not just the
  final PNG
- Deploy as a Streamlit/Gradio app so people can try it without setup
- Add on-screen FPS counter (great addition to show real-time perf)

## Suggested LinkedIn post structure

- 1–2 line hook (what it does + why it's cool)
- Tech stack used
- Bullet list of features (✅ style, matches the reference post)
- What you learned (CV concepts, HCI, real-time processing)
- A ~15–30 sec screen recording of you actually air-writing — this is
  what gets the engagement, more than the text
- Hashtags: #Python #OpenCV #MediaPipe #ComputerVision
  #ArtificialIntelligence #MachineLearning #GestureRecognition
  #ImageProcessing #SoftwareDevelopment #Projects #Learning #Innovation
