from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from projection_math import project_point, transform_vertices


class CubeRenderer:
    def __init__(self) -> None:
        # Unit cube centered at origin.
        self.vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float32,
        )

        # Counter-clockwise winding from outside for back-face culling.
        self.faces: List[Tuple[int, int, int, int]] = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (1, 2, 6, 5),
            (0, 3, 7, 4),
        ]
        self.base_colors = [
            (60, 125, 240),
            (80, 200, 250),
            (70, 160, 255),
            (65, 145, 225),
            (90, 220, 255),
            (50, 110, 215),
        ]

    def render(
        self,
        frame_shape: Sequence[int],
        camera_pos: Tuple[float, float, float],
        object_position: Tuple[float, float, float],
        object_scale: float,
        rotation: Tuple[float, float, float],
        focal_scale: float,
        depth_offset: float,
    ) -> np.ndarray:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        canvas = self._background(h, w)

        cam = np.asarray(camera_pos, dtype=np.float32)
        world_vertices = transform_vertices(
            self.vertices,
            rotation=rotation,
            translation=object_position,
            scale=object_scale,
        )

        draw_queue = []
        light_dir = self._normalize(np.array([0.35, -0.4, -1.0], dtype=np.float32))

        for face_idx, face in enumerate(self.faces):
            pts3 = world_vertices[list(face)]
            normal = np.cross(pts3[1] - pts3[0], pts3[2] - pts3[0])
            if np.linalg.norm(normal) < 1e-8:
                continue

            face_center = np.mean(pts3, axis=0)
            view_vec = cam - face_center

            # Occlusion: back-face culling for hidden faces.
            if float(np.dot(normal, view_vec)) <= 0.0:
                continue

            projected = []
            depth_values = []
            clipped = False
            for v in pts3:
                pr = project_point(
                    point=v,
                    camera_pos=cam,
                    width=w,
                    height=h,
                    focal_scale=focal_scale,
                    depth_offset=depth_offset,
                )
                if pr is None:
                    clipped = True
                    break
                projected.append((pr[0], pr[1]))
                depth_values.append(pr[2])

            if clipped:
                continue

            normal_n = self._normalize(normal)
            brightness = float(np.clip(0.25 + 0.75 * abs(np.dot(normal_n, light_dir)), 0.2, 1.0))
            base = np.array(self.base_colors[face_idx], dtype=np.float32)
            color = tuple(int(c) for c in np.clip(base * brightness, 0, 255))
            avg_depth = float(np.mean(depth_values))
            draw_queue.append((avg_depth, projected, color))

        # Painter's algorithm: far faces first, near faces last.
        draw_queue.sort(key=lambda it: it[0], reverse=True)

        for _, poly, color in draw_queue:
            poly_np = np.array(poly, dtype=np.int32)
            cv2.fillConvexPoly(canvas, poly_np, color, cv2.LINE_AA)
            cv2.polylines(canvas, [poly_np], isClosed=True, color=(25, 25, 25), thickness=1, lineType=cv2.LINE_AA)

        return canvas

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v
        return v / n

    @staticmethod
    def _background(h: int, w: int) -> np.ndarray:
        # Subtle gradient improves depth perception against a flat screen.
        top = np.array([16, 16, 20], dtype=np.float32)
        bot = np.array([32, 32, 38], dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        grad = top * (1.0 - alpha) + bot * alpha
        bg = np.repeat(grad[:, None, :], w, axis=1)
        return bg.astype(np.uint8)
