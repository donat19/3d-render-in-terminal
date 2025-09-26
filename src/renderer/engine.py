"""Core math utilities and rendering engine for terminal 3D graphics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class SceneTriangle:
    v0: Vec3
    edge1: Vec3
    edge2: Vec3
    normal: Vec3
    color: Tuple[int, int, int]
    reflective: float
    triangle_id: int


@dataclass(frozen=True, slots=True)
class HitInfo:
    distance: float
    point: Vec3
    normal: Vec3
    color: Tuple[int, int, int]
    reflective: float
    triangle_id: int


@dataclass(frozen=True, slots=True)
class TraceResult:
    rgb: Tuple[float, float, float]


FrameCell = Tuple[str, Optional[int]]
FrameMatrix = List[List[FrameCell]]


class RenderEngine:
    """Software renderer producing ANSI-coloured frames for terminal output."""

    _GRADIENT = " ░▒▓█"

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
        self._tan_half_fov = math.tan(self._fov_radians / 2.0)
        self.camera_distance = camera_distance
        self.light_direction = light_direction.normalized()
        self.near_clip = near_clip
        self._aspect_ratio = self.width / self.height
        self._max_reflection_depth = 2
        self._ray_cache = None
        self._ray_cache_key = None

    def resize(self, width: int, height: int) -> None:
        if width < 2 or height < 2:
            return
        self.width = width
        self.height = height
        self._aspect_ratio = self.width / self.height
        self._invalidate_ray_cache()

    def set_fov(self, fov_degrees: float) -> None:
        self._fov_degrees = fov_degrees
        self._fov_radians = math.radians(fov_degrees)
        self._tan_half_fov = math.tan(self._fov_radians / 2.0)
        self._invalidate_ray_cache()

    def project_point(self, vertex: Vec3) -> Optional[Tuple[float, float, float]]:
        return self._project(vertex)

    def render(
        self,
        mesh: Mesh,
        rotation: Vec3,
        translation: Vec3 | None = None,
        *,
        floor: Mesh | None = None,
        floor_rotation: Vec3 | None = None,
        floor_translation: Vec3 | None = None,
        cast_shadows: bool = False,
        enable_reflections: bool = True,
        hud: Optional[Sequence[str]] = None,
        hud_color: Optional[int] = 250,
        output_format: str = "ansi",
    ) -> Union[str, FrameMatrix]:
        if self.width < 10 or self.height < 10:
            return (
                "Terminal window too small for rendering. "
                "Resize to at least 10x10 characters.\n"
            )

        if translation is None:
            translation = Vec3(0.0, 0.0, 0.0)
        if floor_rotation is None:
            floor_rotation = Vec3(0.0, 0.0, 0.0)
        if floor_translation is None:
            floor_translation = Vec3(0.0, 0.0, 0.0)

        frame: List[List[Tuple[str, Optional[int]]]] = [
            [(" ", None) for _ in range(self.width)] for _ in range(self.height)
        ]

        scene_triangles = self._build_scene_triangles(
            mesh,
            rotation,
            translation,
            floor,
            floor_rotation,
            floor_translation,
            enable_reflections,
        )

        camera_origin = Vec3(0.0, 0.0, 0.0)
        max_depth = self._max_reflection_depth if enable_reflections else 1

        ray_cache = self._ensure_ray_cache()
        trace_ray = self._trace_ray
        luminance_fn = self._luminance
        char_for_intensity = self._char_for_intensity
        ansi_from_rgb = self._ansi_from_rgb

        for y in range(self.height):
            frame_row = frame[y]
            ray_row = ray_cache[y]
            for x in range(self.width):
                direction = ray_row[x]
                traced = trace_ray(
                    camera_origin,
                    direction,
                    scene_triangles,
                    depth=0,
                    max_depth=max_depth,
                    cast_shadows=cast_shadows,
                )
                if traced is None:
                    continue

                luminance = luminance_fn(traced.rgb)
                if luminance <= 0.02:
                    continue

                char = char_for_intensity(luminance)
                color_code = ansi_from_rgb(traced.rgb)
                frame_row[x] = (char, color_code)

        if hud:
            self._blit_hud(frame, hud, hud_color)

        if output_format == "matrix":
            return frame
        if output_format == "ansi":
            return self._compose_frame(frame)
        raise ValueError(f"Unsupported output_format '{output_format}'")

    # Internal helpers -------------------------------------------------

    def _build_scene_triangles(
        self,
        mesh: Mesh,
        rotation: Vec3,
        translation: Vec3,
        floor: Mesh | None,
        floor_rotation: Vec3,
        floor_translation: Vec3,
        enable_reflections: bool,
    ) -> List[SceneTriangle]:
        scene: List[SceneTriangle] = []
        triangle_id = 0

        def add_mesh(source: Mesh, rot: Vec3, trans: Vec3, reflective: float) -> None:
            nonlocal triangle_id
            for triangle in source:
                v0 = self._transform_vertex(triangle.vertices[0], rot, trans)
                v1 = self._transform_vertex(triangle.vertices[1], rot, trans)
                v2 = self._transform_vertex(triangle.vertices[2], rot, trans)

                if v0.z <= self.near_clip or v1.z <= self.near_clip or v2.z <= self.near_clip:
                    continue

                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = edge1.cross(edge2)
                if normal.length_squared() <= 1e-8:
                    continue
                normal = normal.normalized()

                scene.append(
                    SceneTriangle(
                        v0,
                        edge1,
                        edge2,
                        normal,
                        triangle.base_color,
                        reflective,
                        triangle_id,
                    )
                )
                triangle_id += 1

        add_mesh(mesh, rotation, translation, reflective=0.0)
        if floor is not None:
            reflectivity = 0.55 if enable_reflections else 0.0
            add_mesh(floor, floor_rotation, floor_translation, reflective=reflectivity)

        return scene

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

    def _generate_camera_ray(self, pixel_x: int, pixel_y: int) -> Vec3:
        if self.width <= 0 or self.height <= 0:
            return Vec3(0.0, 0.0, 1.0)

        ndc_x = ((pixel_x + 0.5) / self.width) * 2.0 - 1.0
        ndc_y = 1.0 - ((pixel_y + 0.5) / self.height) * 2.0

        px = ndc_x * self._aspect_ratio * self._tan_half_fov
        py = ndc_y * self._tan_half_fov
        inv_len = 1.0 / math.sqrt(px * px + py * py + 1.0)
        return Vec3(px * inv_len, py * inv_len, inv_len)

    def _ensure_ray_cache(self) -> List[List[Vec3]]:
        key = (self.width, self.height, self._tan_half_fov)
        if self._ray_cache is not None and self._ray_cache_key == key:
            return self._ray_cache

        width, height = self.width, self.height
        rays: List[List[Vec3]] = []
        aspect = self._aspect_ratio
        tan_half_fov = self._tan_half_fov

        for y in range(height):
            row: List[Vec3] = []
            ndc_y = 1.0 - ((y + 0.5) / height) * 2.0
            py = ndc_y * tan_half_fov
            for x in range(width):
                ndc_x = ((x + 0.5) / width) * 2.0 - 1.0
                px = ndc_x * aspect * tan_half_fov
                inv_len = 1.0 / math.sqrt(px * px + py * py + 1.0)
                row.append(Vec3(px * inv_len, py * inv_len, inv_len))
            rays.append(row)

        self._ray_cache = rays
        self._ray_cache_key = key
        return rays

    def _invalidate_ray_cache(self) -> None:
        self._ray_cache = None
        self._ray_cache_key = None

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

    def _trace_ray(
        self,
        origin: Vec3,
        direction: Vec3,
        triangles: Sequence[SceneTriangle],
        *,
        depth: int,
        max_depth: int,
        cast_shadows: bool,
    ) -> Optional[TraceResult]:
        hit = self._closest_intersection(origin, direction, triangles)
        if hit is None:
            return None

        normal = hit.normal
        if normal.dot(direction) > 0:
            normal = -normal

        to_camera = (-direction).normalized()
        in_shadow = cast_shadows and self._is_in_shadow(
            hit.point, normal, triangles, hit.triangle_id
        )
        intensity = self._lighting_intensity(normal, to_camera, in_shadow)

        base_rgb = (
            (hit.color[0] / 5.0) * intensity,
            (hit.color[1] / 5.0) * intensity,
            (hit.color[2] / 5.0) * intensity,
        )

        if hit.reflective > 1e-3 and depth + 1 < max_depth:
            reflection_origin = hit.point + normal * 1e-3
            reflection_direction = self._reflect(direction, normal).normalized()
            reflection = self._trace_ray(
                reflection_origin,
                reflection_direction,
                triangles,
                depth=depth + 1,
                max_depth=max_depth,
                cast_shadows=cast_shadows,
            )
            if reflection is not None:
                blend = hit.reflective
                base_rgb = (
                    (1.0 - blend) * base_rgb[0] + blend * reflection.rgb[0],
                    (1.0 - blend) * base_rgb[1] + blend * reflection.rgb[1],
                    (1.0 - blend) * base_rgb[2] + blend * reflection.rgb[2],
                )

        return TraceResult(self._clamp_rgb(base_rgb))

    def _closest_intersection(
        self,
        origin: Vec3,
        direction: Vec3,
        triangles: Sequence[SceneTriangle],
        exclude_id: Optional[int] = None,
    ) -> Optional[HitInfo]:
        closest_distance = float("inf")
        closest_hit: Optional[HitInfo] = None

        for triangle in triangles:
            if exclude_id is not None and triangle.triangle_id == exclude_id:
                continue

            distance = self._intersect_triangle(origin, direction, triangle)
            if distance is None or distance <= 1e-4 or distance >= closest_distance:
                continue

            point = origin + direction * distance
            closest_distance = distance
            closest_hit = HitInfo(
                distance,
                point,
                triangle.normal,
                triangle.color,
                triangle.reflective,
                triangle.triangle_id,
            )

        return closest_hit

    def _intersect_triangle(
        self, origin: Vec3, direction: Vec3, triangle: SceneTriangle
    ) -> Optional[float]:
        edge1 = triangle.edge1
        edge2 = triangle.edge2
        pvec = direction.cross(edge2)
        det = edge1.dot(pvec)

        if abs(det) <= 1e-8:
            return None

        inv_det = 1.0 / det
        tvec = origin - triangle.v0
        u = tvec.dot(pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None

        qvec = tvec.cross(edge1)
        v = direction.dot(qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return None

        distance = edge2.dot(qvec) * inv_det
        if distance <= 1e-5:
            return None

        return distance

    def _lighting_intensity(
        self, normal: Vec3, to_camera: Vec3, in_shadow: bool
    ) -> float:
        ambient = 0.18
        if in_shadow:
            return ambient

        diffuse = max(0.0, normal.dot(self.light_direction))
        halfway = (self.light_direction + to_camera).normalized()
        specular = 0.0
        if halfway.length_squared() > 1e-8:
            specular = max(0.0, normal.dot(halfway)) ** 24

        intensity = ambient + 0.72 * diffuse + 0.28 * specular
        return max(0.0, min(1.0, intensity))

    def _is_in_shadow(
        self,
        point: Vec3,
        normal: Vec3,
        triangles: Sequence[SceneTriangle],
        triangle_id: int,
    ) -> bool:
        shadow_origin = point + normal * 1e-3
        shadow_direction = -self.light_direction
        hit = self._closest_intersection(
            shadow_origin,
            shadow_direction,
            triangles,
            exclude_id=triangle_id,
        )
        return hit is not None

    @staticmethod
    def _reflect(direction: Vec3, normal: Vec3) -> Vec3:
        return direction - normal * (2.0 * direction.dot(normal))

    def _char_for_intensity(self, intensity: float) -> str:
        normalized = max(0.0, min(1.0, intensity)) ** 0.9
        idx = int(round(normalized * (len(self._GRADIENT) - 1)))
        idx = max(0, min(len(self._GRADIENT) - 1, idx))
        return self._GRADIENT[idx]

    def _ansi_from_rgb(self, rgb: Tuple[float, float, float]) -> int:
        r = int(max(0, min(5, round(rgb[0] * 5))))
        g = int(max(0, min(5, round(rgb[1] * 5))))
        b = int(max(0, min(5, round(rgb[2] * 5))))
        return 16 + 36 * r + 6 * g + b

    @staticmethod
    def _luminance(rgb: Tuple[float, float, float]) -> float:
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    @staticmethod
    def _clamp_rgb(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            max(0.0, min(1.0, rgb[0])),
            max(0.0, min(1.0, rgb[1])),
            max(0.0, min(1.0, rgb[2])),
        )

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

    def _blit_hud(
        self,
        frame: List[List[Tuple[str, Optional[int]]]],
        lines: Sequence[str],
        color: Optional[int],
    ) -> None:
        if not lines:
            return
        max_width = max(len(line) for line in lines)
        start_x = max(0, self.width - max_width - 1)
        for row_offset, line in enumerate(lines):
            if row_offset >= self.height:
                break
            x = start_x
            for char in line:
                if 0 <= x < self.width:
                    frame[row_offset][x] = (char, color)
                x += 1
