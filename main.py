from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from renderer import CubeRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2.5D head-tracked cube illusion")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")

    parser.add_argument("--parallax-x", type=float, default=2.0, help="Horizontal camera offset strength")
    parser.add_argument("--parallax-y", type=float, default=1.4, help="Vertical camera offset strength")
    parser.add_argument("--parallax-z", type=float, default=2.0, help="Depth camera offset strength")

    parser.add_argument("--object-depth", type=float, default=7.0, help="Cube distance from origin")
    parser.add_argument("--object-scale", type=float, default=1.25, help="Cube size")

    parser.add_argument("--focal-scale", type=float, default=0.95, help="Projection focal multiplier")
    parser.add_argument("--depth-offset", type=float, default=1.8, help="Projection denominator offset")

    parser.add_argument("--smoothing", type=float, default=0.22, help="EMA smoothing alpha")
    parser.add_argument("--deadzone-xy", type=float, default=0.015, help="Dead-zone for x/y")
    parser.add_argument("--deadzone-z", type=float, default=0.01, help="Dead-zone for z")

    parser.add_argument("--rotation-speed", type=float, default=0.8, help="Cube spin speed (rad/s)")
    parser.add_argument("--debug", action="store_true", help="Show webcam + tracking debug overlay")
    return parser.parse_args()


def overlay_preview(canvas: np.ndarray, preview: np.ndarray, padding: int = 16) -> None:
    target_w = int(canvas.shape[1] * 0.24)
    scale = target_w / preview.shape[1]
    target_h = int(preview.shape[0] * scale)
    if target_h + padding * 2 > canvas.shape[0]:
        target_h = canvas.shape[0] - padding * 2
        scale = target_h / preview.shape[0]
        target_w = int(preview.shape[1] * scale)

    thumb = cv2.resize(preview, (target_w, target_h), interpolation=cv2.INTER_AREA)
    y0, y1 = padding, padding + target_h
    x0, x1 = padding, padding + target_w

    canvas[y0:y1, x0:x1] = thumb
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (230, 230, 230), 1, cv2.LINE_AA)


def main() -> None:
    try:
        from camera_tracking import HeadTracker
    except Exception as exc:
        raise SystemExit(
            "Failed to initialize tracking dependencies. "
            "Install requirements and use Python 3.10-3.12 in your virtual environment. "
            f"Details: {exc}"
        ) from exc

    if sys.platform == "win32" and sys.maxsize < 2**32:
        raise SystemExit("64-bit Python is required on Windows.")

    args = parse_args()

    tracker = HeadTracker(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        smoothing_alpha=args.smoothing,
        deadzone_xy=args.deadzone_xy,
        deadzone_z=args.deadzone_z,
    )
    renderer = CubeRenderer()

    window_name = "2.5D Head-Tracked Cube"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_t = time.time()
    angle = 0.0
    fps = 0.0

    try:
        while True:
            frame, state = tracker.read()
            if frame is None:
                break

            now = time.time()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            fps = 0.92 * fps + 0.08 * (1.0 / dt)
            angle += args.rotation_speed * dt

            cam_x = state.x * args.parallax_x
            cam_y = -state.y * args.parallax_y
            cam_z = state.z * args.parallax_z

            canvas = renderer.render(
                frame_shape=frame.shape,
                camera_pos=(cam_x, cam_y, cam_z),
                object_position=(0.0, 0.0, args.object_depth),
                object_scale=args.object_scale,
                rotation=(angle * 0.4, angle, angle * 0.2),
                focal_scale=args.focal_scale,
                depth_offset=args.depth_offset,
            )

            cv2.putText(
                canvas,
                f"FPS: {fps:5.1f}",
                (12, canvas.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            if args.debug:
                preview = tracker.draw_debug(frame.copy(), state)
                overlay_preview(canvas, preview)
                cv2.putText(
                    canvas,
                    f"virtual cam xyz: {cam_x:+.2f} {cam_y:+.2f} {cam_z:+.2f}",
                    (12, canvas.shape[0] - 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
