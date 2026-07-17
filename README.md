# Virtual Air Writing System ✍️

I Built A real-time computer vision app that lets you "write" in the air using
hand gestures captured through webcam — no touch screen or stylus
required.

Built with *Python*, *OpenCV*, and *MediaPipe*.

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
| ☝️ Only index finger up | **Draw mode** — moves the brush and draws a line 
| ✌️ Index + middle finger up | **Select mode** — hover over the toolbar to pick a color / eraser / clear, without drawing 
| ✊ Fist / other fingers up | **Pen up** — stops drawing (like lifting a pen off paper) 

The toolbar sits at the top of the frame. Point at a color swatch, the
eraser button, or the clear button in select mode to activate it.

Keyboard shortcuts (handy for demos or if gesture detection misfires):
- `s` — save the current canvas to `saved_drawings/`
- `c` — clear the canvas
- `q` or `Esc` — quit
