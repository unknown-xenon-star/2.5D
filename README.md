# 2.5D Head-Tracked Cube (Python)

This app creates a 2.5D fish-tank VR illusion: a cube is rendered in a 2D window, and its perspective changes in real time from webcam-based head tracking.

## Features

- Real-time webcam face/eye tracking (MediaPipe Face Mesh)
- Stabilized head pose `(x, y, z)` with EMA smoothing + dead-zone
- Manual virtual camera and custom perspective projection
- 3D cube represented by local-space vertices
- Back-face culling + painter sorting for occlusion
- Optional debug overlay with landmarks and vectors

## Project Structure

- `camera_tracking.py`: webcam capture + landmarks + stabilized head motion
- `projection_math.py`: transform and perspective math
- `renderer.py`: 2.5D cube rendering and occlusion handling
- `main.py`: runtime loop and parameter controls

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --debug
```

Press `q` or `Esc` to quit.

## Useful Parameters

```bash
python main.py \
  --parallax-x 2.2 \
  --parallax-y 1.6 \
  --parallax-z 2.3 \
  --focal-scale 0.95 \
  --depth-offset 1.8 \
  --rotation-speed 0.8 \
  --debug
```

## Calibration Tips

- Sit centered in front of the camera for 1-2 seconds at startup (for depth baseline).
- Increase `--parallax-x/--parallax-y` for stronger side/up-down perspective response.
- Increase `--parallax-z` for stronger near/far depth scaling.
- If jitter is visible, increase `--smoothing` slightly and/or raise dead-zones.

## Notes

- Requires a working webcam.
- Uses OpenCV rendering only; no full 3D engine is used.
