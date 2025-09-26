"""Core math utilities and rendering engine for terminal 3D graphics."""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
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
        max_reflection_depth: int = 2,
        sampling_step: int = 1,
        ambient_strength: float = 0.2,
        sun_intensity: float = 1.0,
        sun_color: Tuple[float, float, float] = (1.0, 0.97, 0.85),
        sun_diffuse_strength: float = 0.85,
        sun_specular_strength: float = 0.3,
        sun_specular_exponent: float = 32.0,
        sun_angular_radius_degrees: float = 0.53,
        gi_strength: float = 0.35,
        sky_color_top: Tuple[float, float, float] = (0.25, 0.45, 0.85),
        sky_color_horizon: Tuple[float, float, float] = (0.85, 0.78, 0.62),
        ground_color: Tuple[float, float, float] = (0.35, 0.30, 0.28),
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
        self._max_reflection_depth = max(0, int(max_reflection_depth))
        self._sampling_step = max(1, int(sampling_step))
        self._ambient_strength = max(0.0, ambient_strength)
        self._sun_intensity = max(0.0, sun_intensity)
        self._sun_color = self._clamp_rgb(sun_color)
        self._sun_diffuse_strength = max(0.0, sun_diffuse_strength)
        self._sun_specular_strength = max(0.0, sun_specular_strength)
        self._sun_specular_exponent = max(1.0, sun_specular_exponent)
        self._gi_strength = max(0.0, gi_strength)
        self._sky_color_top = self._clamp_rgb(sky_color_top)
        self._sky_color_horizon = self._clamp_rgb(sky_color_horizon)
        self._ground_color = self._clamp_rgb(ground_color)
        sun_radius = max(0.05, sun_angular_radius_degrees)
        self._sun_cos_core = math.cos(math.radians(sun_radius))
        self._sun_cos_halo = math.cos(math.radians(min(90.0, sun_radius * 3.0)))
        if self._sun_cos_halo > self._sun_cos_core:
            self._sun_cos_halo, self._sun_cos_core = self._sun_cos_core, self._sun_cos_halo
        self._up_vector = Vec3(0.0, 1.0, 0.0)
        self._horizon_haze_strength = 0.5
        self._ray_cache = None
        self._ray_cache_key = None
        self._async_chunk_rows = 16

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

    def set_max_reflection_depth(self, depth: int) -> None:
        self._max_reflection_depth = max(0, int(depth))

    def set_sampling_step(self, step: int) -> None:
        self._sampling_step = max(1, int(step))

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

        frame, scene_triangles, camera_origin, max_depth = self._prepare_render_inputs(
            mesh,
            rotation,
            translation,
            floor,
            floor_rotation,
            floor_translation,
            enable_reflections,
        )

        ray_cache = self._ensure_ray_cache()
        _, chunk_rows = self._compute_chunk(
            self.width,
            scene_triangles,
            ray_cache,
            camera_origin,
            max_depth,
            cast_shadows,
            0,
            self.height,
        )
        for row_index, row in enumerate(chunk_rows):
            frame[row_index] = row

        if hud:
            self._blit_hud(frame, hud, hud_color)

        if output_format == "matrix":
            return frame
        if output_format == "ansi":
            return self._compose_frame(frame)
        raise ValueError(f"Unsupported output_format '{output_format}'")

    async def render_async(
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
        executor: ThreadPoolExecutor | None = None,
        chunk_rows: int | None = None,
    ) -> Union[str, FrameMatrix]:
        if self.width < 10 or self.height < 10:
            return (
                "Terminal window too small for rendering. "
                "Resize to at least 10x10 characters.\n"
            )

        frame, scene_triangles, camera_origin, max_depth = self._prepare_render_inputs(
            mesh,
            rotation,
            translation,
            floor,
            floor_rotation,
            floor_translation,
            enable_reflections,
        )

        ray_cache = self._ensure_ray_cache()
        height = self.height
        width = self.width
        sampling_step = self._sampling_step
        chunk_size = chunk_rows if chunk_rows is not None else self._async_chunk_rows
        chunk_size = max(sampling_step, int(chunk_size))

        loop = asyncio.get_running_loop()
        local_executor = executor
        created_executor = False
        if local_executor is None:
            local_executor = ThreadPoolExecutor(max_workers=4)
            created_executor = True

        try:
            tasks = [
                loop.run_in_executor(
                    local_executor,
                    self._compute_chunk,
                    width,
                    scene_triangles,
                    ray_cache,
                    camera_origin,
                    max_depth,
                    cast_shadows,
                    y_start,
                    min(height, y_start + chunk_size),
                )
                for y_start in range(0, height, chunk_size)
            ]
            results = await asyncio.gather(*tasks)
        finally:
            if created_executor and local_executor is not None:
                local_executor.shutdown(wait=True)

        results.sort(key=lambda item: item[0])
        for y_start, chunk_rows_data in results:
            for offset, row in enumerate(chunk_rows_data):
                frame[y_start + offset] = row

        if hud:
            self._blit_hud(frame, hud, hud_color)

        if output_format == "matrix":
            return frame
        if output_format == "ansi":
            return self._compose_frame(frame)
        raise ValueError(f"Unsupported output_format '{output_format}'")

    # Internal helpers -------------------------------------------------

    def _prepare_render_inputs(
        self,
        mesh: Mesh,
        rotation: Vec3,
        translation: Vec3 | None,
        floor: Mesh | None,
        floor_rotation: Vec3 | None,
        floor_translation: Vec3 | None,
        enable_reflections: bool,
    ) -> Tuple[List[List[FrameCell]], List[SceneTriangle], Vec3, int]:
        if translation is None:
            translation = Vec3(0.0, 0.0, 0.0)
        if floor_rotation is None:
            floor_rotation = Vec3(0.0, 0.0, 0.0)
        if floor_translation is None:
            floor_translation = Vec3(0.0, 0.0, 0.0)

        frame: List[List[FrameCell]] = [
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
        return frame, scene_triangles, camera_origin, max_depth

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

    def _compute_chunk(
        self,
        frame_width: int,
        scene_triangles: Sequence[SceneTriangle],
        ray_cache: Sequence[Sequence[Vec3]],
        camera_origin: Vec3,
        max_depth: int,
        cast_shadows: bool,
        y_start: int,
        y_end: int,
    ) -> Tuple[int, List[List[FrameCell]]]:
        sampling_step = self._sampling_step
        luminance_fn = self._luminance
        char_for_intensity = self._char_for_intensity
        ansi_from_rgb = self._ansi_from_rgb
        trace_ray = self._trace_ray

        chunk_height = max(0, y_end - y_start)
        chunk_frame: List[List[FrameCell]] = [
            [(" ", None) for _ in range(frame_width)] for _ in range(chunk_height)
        ]

        for y in range(y_start, y_end, sampling_step):
            ray_row = ray_cache[y]
            for x in range(0, frame_width, sampling_step):
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
                block_y_end = min(y_end, y + sampling_step)
                block_x_end = min(frame_width, x + sampling_step)
                for yy in range(y, block_y_end):
                    local_row = chunk_frame[yy - y_start]
                    for xx in range(x, block_x_end):
                        local_row[xx] = (char, color_code)

        return y_start, chunk_frame

    def _environment_color(self, direction: Vec3) -> Tuple[float, float, float]:
        y = max(-1.0, min(1.0, direction.y))
        abs_y = abs(y)
        horizon_mix = (1.0 - abs_y) ** 2

        if y >= 0.0:
            sky_t = y ** 0.35
            base = self._lerp_rgb(self._sky_color_horizon, self._sky_color_top, sky_t)
        else:
            ground_t = (-y) ** 0.45
            darker_ground = (
                self._ground_color[0] * 0.35,
                self._ground_color[1] * 0.35,
                self._ground_color[2] * 0.35,
            )
            base = self._lerp_rgb(self._ground_color, darker_ground, ground_t)

        if horizon_mix > 0.0:
            base = self._lerp_rgb(
                base,
                self._sky_color_horizon,
                min(1.0, horizon_mix * self._horizon_haze_strength),
            )

        cos_angle = direction.dot(self.light_direction)
        sun_factor = 0.0
        if cos_angle >= self._sun_cos_core:
            sun_factor = 1.0
        elif cos_angle >= self._sun_cos_halo:
            denom = self._sun_cos_core - self._sun_cos_halo
            if denom > 1e-6:
                sun_factor = (cos_angle - self._sun_cos_halo) / denom
                sun_factor *= sun_factor

        if sun_factor > 0.0:
            sun_strength = self._sun_intensity * (0.35 + 0.65 * sun_factor)
            base = (
                base[0] + self._sun_color[0] * sun_strength,
                base[1] + self._sun_color[1] * sun_strength,
                base[2] + self._sun_color[2] * sun_strength,
            )

        return self._clamp_rgb(base)

    def _global_illumination_contribution(self, normal: Vec3) -> Tuple[float, float, float]:
        if self._gi_strength <= 1e-6:
            return (0.0, 0.0, 0.0)

        n = normal.normalized()
        up_dot = max(0.0, n.dot(self._up_vector))
        down_dot = max(0.0, n.dot(-self._up_vector))
        horizon = max(0.0, 1.0 - abs(n.y))

        sky_rgb = self._lerp_rgb(self._sky_color_horizon, self._sky_color_top, up_dot ** 0.5)
        ground_rgb = self._ground_color
        horizon_rgb = self._sky_color_horizon

        bounce = (
            sky_rgb[0] * up_dot + horizon_rgb[0] * horizon * 0.25 + ground_rgb[0] * down_dot * 0.6,
            sky_rgb[1] * up_dot + horizon_rgb[1] * horizon * 0.25 + ground_rgb[1] * down_dot * 0.6,
            sky_rgb[2] * up_dot + horizon_rgb[2] * horizon * 0.25 + ground_rgb[2] * down_dot * 0.6,
        )

        strength = self._gi_strength
        return (
            bounce[0] * strength,
            bounce[1] * strength,
            bounce[2] * strength,
        )

    def _invalidate_ray_cache(self) -> None:
        self._ray_cache = None
        self._ray_cache_key = None

    @staticmethod
    def _lerp_rgb(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
        t = max(0.0, min(1.0, t))
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )

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
            return TraceResult(self._environment_color(direction))

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

        if self._gi_strength > 1e-6:
            gi_rgb = self._global_illumination_contribution(normal)
            base_rgb = (
                base_rgb[0] + gi_rgb[0],
                base_rgb[1] + gi_rgb[1],
                base_rgb[2] + gi_rgb[2],
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
        ambient = self._ambient_strength

        if in_shadow:
            return min(1.0, ambient)

        diffuse = max(0.0, normal.dot(self.light_direction))
        halfway = (self.light_direction + to_camera).normalized()
        specular = 0.0
        if halfway.length_squared() > 1e-8:
            specular = max(0.0, normal.dot(halfway)) ** self._sun_specular_exponent

        intensity = ambient
        intensity += self._sun_diffuse_strength * diffuse
        intensity += self._sun_specular_strength * specular

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
