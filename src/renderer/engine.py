"""Core math utilities and rendering engine for terminal 3D graphics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Vec3:
    """Lightweight immutable 3D vector."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        if not isinstance(scalar, (int, float)):
            raise TypeError("Vec3 can only be multiplied by a scalar")
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        if scalar == 0:
            raise ZeroDivisionError("Division by zero in Vec3")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def length_squared(self) -> float:
        return self.dot(self)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length <= 1e-8:
            return Vec3(0.0, 0.0, 0.0)
        return self / length


@dataclass(frozen=True)
class Triangle:
    """A triangle with a fixed base color in ANSI 256-colour RGB cube coordinates."""

    vertices: Tuple[Vec3, Vec3, Vec3]
    base_color: Tuple[int, int, int]


class Mesh:
    """Collection of shaded triangles."""

    def __init__(self, triangles: Sequence[Triangle]):
        self._triangles: Tuple[Triangle, ...] = tuple(triangles)
        if not self._triangles:
            raise ValueError("Mesh requires at least one triangle")

    @property
    def triangles(self) -> Tuple[Triangle, ...]:
        return self._triangles

    def __iter__(self):
        return iter(self._triangles)


class RenderEngine:
    """Software renderer producing ANSI-coloured frames for terminal output."""

    _GRADIENT = " .:-=+*#%@"

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fov_degrees: float = 70.0,
        camera_distance: float = 5.0,
        light_direction: Vec3 = Vec3(-0.4, 0.8, -0.6),
        near_clip: float = 0.1,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError("RenderEngine requires width and height >= 2")
        self.width = width
        self.height = height
        self._fov_degrees = fov_degrees
        self._fov_radians = math.radians(fov_degrees)
        self.camera_distance = camera_distance
        self.light_direction = light_direction.normalized()
        self.near_clip = near_clip
        self._aspect_ratio = self.width / self.height

    def resize(self, width: int, height: int) -> None:
        if width < 2 or height < 2:
            return
        self.width = width
        self.height = height
        self._aspect_ratio = self.width / self.height

    def set_fov(self, fov_degrees: float) -> None:
        self._fov_degrees = fov_degrees
        self._fov_radians = math.radians(fov_degrees)

    def project_point(self, vertex: Vec3) -> Optional[Tuple[float, float, float]]:
        return self._project(vertex)

    def render(
        self,
        mesh: Mesh,
        rotation: Vec3,
        translation: Vec3 | None = None,
    ) -> str:
        if self.width < 10 or self.height < 10:
            return (
                "Terminal window too small for rendering. "
                "Resize to at least 10x10 characters.\n"
            )

        if translation is None:
            translation = Vec3(0.0, 0.0, 0.0)

        depth_buffer: List[List[float]] = [
            [float("inf")] * self.width for _ in range(self.height)
        ]
        frame: List[List[Tuple[str, Optional[int]]]] = [
            [(" ", None) for _ in range(self.width)] for _ in range(self.height)
        ]

        for triangle in mesh:
            transformed = [
                self._transform_vertex(vertex, rotation, translation)
                for vertex in triangle.vertices
            ]

            if any(vertex.z <= self.near_clip for vertex in transformed):
                continue

            normal = (transformed[1] - transformed[0]).cross(
                transformed[2] - transformed[0]
            )
            if normal.length_squared() <= 1e-8:
                continue
            normal = normal.normalized()

            centroid = (
                transformed[0] + transformed[1] + transformed[2]
            ) * (1.0 / 3.0)
            to_camera = (-centroid).normalized()
            if normal.dot(to_camera) <= 0:
                continue

            illumination = max(0.12, min(1.0, normal.dot(self.light_direction)))

            projected: List[Tuple[float, float, float]] = []
            skip_triangle = False
            for vertex in transformed:
                projected_vertex = self._project(vertex)
                if projected_vertex is None:
                    skip_triangle = True
                    break
                projected.append(projected_vertex)

            if skip_triangle:
                continue

            self._rasterize_triangle(
                projected,
                transformed,
                triangle.base_color,
                illumination,
                frame,
                depth_buffer,
            )

        return self._compose_frame(frame)

    # Internal helpers -------------------------------------------------

    def _transform_vertex(
        self, vertex: Vec3, rotation: Vec3, translation: Vec3
    ) -> Vec3:
        rotated = self._apply_rotation(vertex, rotation)
        shifted = rotated + translation + Vec3(0.0, 0.0, self.camera_distance)
        return shifted

    def _apply_rotation(self, vertex: Vec3, rotation: Vec3) -> Vec3:
        rx, ry, rz = rotation.x, rotation.y, rotation.z

        cos_rx, sin_rx = math.cos(rx), math.sin(rx)
        y = vertex.y * cos_rx - vertex.z * sin_rx
        z = vertex.y * sin_rx + vertex.z * cos_rx
        rotated_x = Vec3(vertex.x, y, z)

        cos_ry, sin_ry = math.cos(ry), math.sin(ry)
        x = rotated_x.x * cos_ry + rotated_x.z * sin_ry
        z = -rotated_x.x * sin_ry + rotated_x.z * cos_ry
        rotated_y = Vec3(x, rotated_x.y, z)

        cos_rz, sin_rz = math.cos(rz), math.sin(rz)
        x = rotated_y.x * cos_rz - rotated_y.y * sin_rz
        y = rotated_y.x * sin_rz + rotated_y.y * cos_rz
        return Vec3(x, y, rotated_y.z)

    def _project(self, vertex: Vec3) -> Optional[Tuple[float, float, float]]:
        if vertex.z <= self.near_clip:
            return None
        aspect = self._aspect_ratio
        f = 1.0 / math.tan(self._fov_radians / 2.0)
        x_ndc = (vertex.x * f) / (vertex.z * aspect)
        y_ndc = (vertex.y * f) / vertex.z

        x_screen = (x_ndc + 1.0) * 0.5 * (self.width - 1)
        y_screen = (1.0 - (y_ndc + 1.0) * 0.5) * (self.height - 1)
        return (x_screen, y_screen, vertex.z)

    def _rasterize_triangle(
        self,
        projected: Sequence[Tuple[float, float, float]],
        transformed: Sequence[Vec3],
        base_color: Tuple[int, int, int],
        illumination: float,
        frame: List[List[Tuple[str, Optional[int]]]],
        depth_buffer: List[List[float]],
    ) -> None:
        p0, p1, p2 = projected
        x_coords = [p0[0], p1[0], p2[0]]
        y_coords = [p0[1], p1[1], p2[1]]

        min_x = max(0, int(math.floor(min(x_coords))))
        max_x = min(self.width - 1, int(math.ceil(max(x_coords))))
        min_y = max(0, int(math.floor(min(y_coords))))
        max_y = min(self.height - 1, int(math.ceil(max(y_coords))))

        if min_x >= max_x or min_y >= max_y:
            return

        def edge(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

        area = edge(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
        if abs(area) <= 1e-8:
            return

        z_values = [p0[2], p1[2], p2[2]]

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px = x + 0.5
                py = y + 0.5
                w0 = edge(p1[0], p1[1], p2[0], p2[1], px, py) / area
                w1 = edge(p2[0], p2[1], p0[0], p0[1], px, py) / area
                w2 = edge(p0[0], p0[1], p1[0], p1[1], px, py) / area

                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                    continue

                depth = w0 * z_values[0] + w1 * z_values[1] + w2 * z_values[2]
                if depth >= depth_buffer[y][x]:
                    continue

                depth_buffer[y][x] = depth
                char = self._char_for_intensity(illumination)
                color = self._color_for_intensity(base_color, illumination)
                frame[y][x] = (char, color)

    def _char_for_intensity(self, intensity: float) -> str:
        idx = int(max(0, min(len(self._GRADIENT) - 1, round(intensity * (len(self._GRADIENT) - 1)))))
        return self._GRADIENT[idx]

    def _color_for_intensity(
        self, base_color: Tuple[int, int, int], intensity: float
    ) -> int:
        r, g, b = base_color
        def clamp(value: float) -> int:
            return int(max(0, min(5, round(value))))
        scaled_r = clamp(r * intensity)
        scaled_g = clamp(g * intensity)
        scaled_b = clamp(b * intensity)
        return 16 + 36 * scaled_r + 6 * scaled_g + scaled_b

    def _compose_frame(
        self, frame: Sequence[Sequence[Tuple[str, Optional[int]]]]
    ) -> str:
        reset = "\033[0m"
        lines: List[str] = []
        for row in frame:
            current_color: Optional[int] = None
            parts: List[str] = []
            for char, color in row:
                if color != current_color:
                    if color is None:
                        parts.append(reset)
                    else:
                        parts.append(f"\033[38;5;{color}m")
                    current_color = color
                parts.append(char)
            if current_color is not None:
                parts.append(reset)
            lines.append("".join(parts))
        return "\n".join(lines)
