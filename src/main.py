"""Interactive entry point for the terminal 3D renderer demo."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, cast

from .renderer.engine import FrameMatrix, Mesh, RenderEngine, Vec3
from .renderer.objects import cornell_box_mesh, cube_mesh, floor_mesh
from .renderer.terminal import TerminalController


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colour 3D renderer for your terminal")
    parser.add_argument("--fps", type=float, default=20.0, help="Target frames per second (default: 20)")
    parser.add_argument("--fov", type=float, default=70.0, help="Field of view in degrees (default: 70)")
    parser.add_argument(
        "--distance",
        type=float,
        default=5.0,
        help="Distance from camera to mesh centre (default: 5)",
    )
    parser.add_argument(
        "--light",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-0.4, 0.8, -0.6),
        help="Directional light vector components",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="cube",
        choices=["cube", "cornell"],
        help="Which demo object to render",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Uniform scale multiplier applied to the mesh",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Multiplier for the rotation speed",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Run for a fixed number of frames (0 = infinite)",
    )
    parser.add_argument(
        "--no-floor",
        action="store_true",
        help="Disable rendering the checkerboard ground plane",
    )
    parser.add_argument(
        "--no-shadows",
        action="store_true",
        help="Disable shadow casting onto the floor",
    )
    parser.add_argument(
        "--no-reflections",
        action="store_true",
        help="Disable reflective floor rendering",
    )
    parser.add_argument(
        "--floor-size",
        type=float,
        default=12.0,
        help="Edge length of the floor plane in world units",
    )
    parser.add_argument(
        "--floor-tiles",
        type=int,
        default=10,
        help="Number of checkerboard tiles per side",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use experimental DRM/KMS presenter instead of ANSI output",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Enable asynchronous frame computation (optimised for ARM CPUs)",
    )
    parser.add_argument(
        "--pixel-step",
        type=int,
        default=1,
        help="Render every Nth pixel and fill neighbouring cells (default: 1)",
    )
    parser.add_argument(
        "--reflection-depth",
        type=int,
        default=2,
        help="Maximum number of recursive reflection bounces (default: 2)",
    )
    parser.add_argument(
        "--ambient",
        type=float,
        default=0.22,
        help="Ambient light strength (default: 0.22)",
    )
    parser.add_argument(
        "--gi-strength",
        type=float,
        default=0.35,
        help="Sky/ground global illumination strength (default: 0.35)",
    )
    parser.add_argument(
        "--sun-intensity",
        type=float,
        default=1.4,
        help="Sun disk intensity multiplier (default: 1.4)",
    )
    return parser.parse_args()


def _mesh_factory(name: str, scale: float) -> Mesh:
    factories: dict[str, Callable[[float], Mesh]] = {
        "cube": cube_mesh,
        "cornell": lambda _scale: cornell_box_mesh(),
    }
    try:
        factory = factories[name]
    except KeyError as exc:  # pragma: no cover - safeguarded by argparse choices
        raise ValueError(f"Unknown mesh '{name}'") from exc
    return factory(scale)


def _ensure_light_vector(vector: tuple[float, float, float]) -> Vec3:
    vec = Vec3(*vector)
    normalised = vec.normalized()
    if normalised.length_squared() <= 1e-8:
        return Vec3(-0.4, 0.8, -0.6).normalized()
    return normalised


def _is_arm_platform() -> bool:
    machine = platform.machine().lower()
    return any(token in machine for token in ("arm", "aarch64", "arm64"))


@dataclass
class RuntimeConfig:
    controller: Optional[TerminalController]
    display_context: Any
    gpu_display: Optional[Any]
    using_gpu: bool
    warnings: list[str]
    fps: float
    frame_duration: float
    mesh: Mesh
    floor: Optional[Mesh]
    floor_translation: Vec3
    floor_rotation: Vec3
    cast_shadows: bool
    enable_reflections: bool
    object_translation: Vec3
    camera_distance: float
    light: Vec3
    is_cornell: bool
    pixel_step: int
    reflection_depth: int
    ambient: float
    gi_strength: float
    sun_intensity: float
    initial_rotation: Vec3
    rotation_velocity: Vec3
    orbit_step: float
    orbit_pitch_limit: float
    async_mode: bool
    async_workers: int
    async_chunk_rows: int


def _setup_runtime(args: argparse.Namespace) -> RuntimeConfig:
    warnings: list[str] = []
    is_cornell = args.object == "cornell"

    default_light = (-0.4, 0.8, -0.6)
    requested_light = tuple(args.light)
    if is_cornell and requested_light == default_light:
        requested_light = (0.2, -1.0, -0.2)
    light = _ensure_light_vector(requested_light)

    controller: Optional[TerminalController] = None
    display_context: Any | None = None
    gpu_display: Any | None = None
    using_gpu = False

    if args.gpu:
        try:
            from .renderer.drm import DrmDisplay

            gpu_display = DrmDisplay()
            if gpu_display.is_ready:
                display_context = gpu_display
                using_gpu = True
            else:
                failure = gpu_display.failure_reason or "GPU backend unavailable"
                warnings.append(f"GPU backend disabled: {failure}")
        except Exception as exc:  # pragma: no cover - environment specific
            warnings.append(f"GPU backend initialisation failed: {exc}")

    if display_context is None:
        controller = TerminalController()
        display_context = controller

    fps = max(1.0, args.fps)
    frame_duration = 1.0 / fps
    mesh = _mesh_factory(args.object, args.scale)

    floor: Optional[Mesh] = None
    floor_translation = Vec3(0.0, -0.5 * args.scale - 1.1, 0.0)
    cast_shadows = False
    enable_reflections = False
    object_translation = Vec3(0.0, 0.0, 0.0)
    floor_rotation = Vec3(0.0, 0.0, 0.0)

    if not is_cornell:
        floor = None if args.no_floor else floor_mesh(args.floor_size, tiles=max(1, args.floor_tiles))
        cast_shadows = floor is not None and not args.no_shadows
        enable_reflections = floor is not None and not args.no_reflections
    else:
        object_translation = Vec3(0.0, -0.25, -1.0)

    camera_distance = args.distance
    if is_cornell and camera_distance < 6.0:
        camera_distance = 6.0

    pixel_step = max(1, args.pixel_step)
    reflection_depth = max(0, args.reflection_depth)

    if using_gpu and args.pixel_step == 1:
        pixel_step = 2
        warnings.append("GPU mode auto-adjust: pixel-step set to 2 for performance")
    if using_gpu and args.reflection_depth > 1:
        reflection_depth = 1
        warnings.append("GPU mode auto-adjust: reflection depth capped to 1 for performance")

    ambient = max(0.0, args.ambient)
    gi_strength = max(0.0, args.gi_strength)
    sun_intensity = max(0.0, args.sun_intensity)

    if is_cornell:
        initial_rotation = Vec3(0.1, 0.3, 0.0)
        rotation_velocity = Vec3(0.0, 0.0, 0.0)
    else:
        initial_rotation = Vec3(0.0, 0.0, 0.0)
        rotation_velocity = Vec3(0.9, 1.1, 0.6) * args.speed

    orbit_step = math.radians(3.0)
    orbit_pitch_limit = math.radians(65.0)

    async_mode = bool(getattr(args, "async_mode", False))
    cpu_count = os.cpu_count() or 2
    is_arm = _is_arm_platform()
    async_workers = max(2, cpu_count - 1 if is_arm else min(cpu_count, 4))
    async_chunk_rows = max(8, pixel_step * 4)

    if async_mode and not is_arm:
        warnings.append("Async mode requested on non-ARM CPU; performance gains may be limited")

    return RuntimeConfig(
        controller=controller,
        display_context=display_context,
        gpu_display=gpu_display,
        using_gpu=using_gpu,
        warnings=warnings,
        fps=fps,
        frame_duration=frame_duration,
        mesh=mesh,
        floor=floor,
        floor_translation=floor_translation,
        floor_rotation=floor_rotation,
        cast_shadows=cast_shadows,
        enable_reflections=enable_reflections,
        object_translation=object_translation,
        camera_distance=camera_distance,
        light=light,
        is_cornell=is_cornell,
        pixel_step=pixel_step,
        reflection_depth=reflection_depth,
        ambient=ambient,
        gi_strength=gi_strength,
        sun_intensity=sun_intensity,
        initial_rotation=initial_rotation,
        rotation_velocity=rotation_velocity,
        orbit_step=orbit_step,
        orbit_pitch_limit=orbit_pitch_limit,
        async_mode=async_mode,
        async_workers=async_workers,
        async_chunk_rows=async_chunk_rows,
    )


def _emit_warnings(warnings: Sequence[str]) -> None:
    if not warnings:
        return
    for warning in warnings:
        sys.stderr.write(f"[renderer] {warning}\n")
    sys.stderr.flush()


def _create_engine(width: int, height: int, args: argparse.Namespace, config: RuntimeConfig) -> RenderEngine:
    engine = RenderEngine(
        width,
        height,
        fov_degrees=args.fov,
        camera_distance=config.camera_distance,
        light_direction=config.light,
        max_reflection_depth=config.reflection_depth,
        sampling_step=config.pixel_step,
        ambient_strength=config.ambient,
        sun_intensity=config.sun_intensity,
        gi_strength=config.gi_strength,
    )
    if config.is_cornell and args.fov == 70.0:
        engine.set_fov(50.0)
    return engine


def _update_rotation(rotation: Vec3, velocity: Vec3, delta: float) -> Vec3:
    if velocity.length_squared() <= 0.0:
        return rotation
    return Vec3(
        rotation.x + velocity.x * delta,
        rotation.y + velocity.y * delta,
        rotation.z + velocity.z * delta,
    )


def _run_sync_loop(args: argparse.Namespace, config: RuntimeConfig) -> None:
    controller = config.controller
    display_context = config.display_context
    using_gpu = config.using_gpu
    gpu_display = config.gpu_display

    with display_context as active_display:
        width, height = active_display.size_tuple()
        engine = _create_engine(width, height, args, config)

        rotation = config.initial_rotation
        rotation_velocity = config.rotation_velocity
        orbit_pitch = 0.0
        orbit_yaw = 0.0
        orbit_step = config.orbit_step
        orbit_pitch_limit = config.orbit_pitch_limit

        frame_counter = 0
        last_frame_start: float | None = None
        smoothed_fps = max(1.0, args.fps)

        try:
            while True:
                frame_start = time.perf_counter()
                delta = 0.0
                if last_frame_start is not None:
                    delta = frame_start - last_frame_start
                    instantaneous_fps = 1.0 / max(delta, 1e-6)
                    smoothed_fps = smoothed_fps * 0.85 + instantaneous_fps * 0.15
                last_frame_start = frame_start

                if not using_gpu:
                    width, height = active_display.size_tuple()
                    engine.resize(width, height)

                rotation = _update_rotation(rotation, rotation_velocity, delta)

                if not config.is_cornell and controller is not None:
                    for key in controller.poll_keys():
                        if key == "LEFT":
                            orbit_yaw += orbit_step
                        elif key == "RIGHT":
                            orbit_yaw -= orbit_step
                        elif key == "UP":
                            orbit_pitch = min(orbit_pitch + orbit_step, orbit_pitch_limit)
                        elif key == "DOWN":
                            orbit_pitch = max(orbit_pitch - orbit_step, -orbit_pitch_limit)

                    if orbit_yaw > math.pi:
                        orbit_yaw -= math.tau
                    elif orbit_yaw < -math.pi:
                        orbit_yaw += math.tau

                orbit_rotation = Vec3(orbit_pitch, orbit_yaw, 0.0) if not config.is_cornell else Vec3(0.0, 0.0, 0.0)
                combined_rotation = Vec3(
                    rotation.x + orbit_rotation.x,
                    rotation.y + orbit_rotation.y,
                    rotation.z + orbit_rotation.z,
                )

                hud_lines = (f"FPS {smoothed_fps:5.1f}",)
                if not config.is_cornell:
                    hud_lines = hud_lines + ("Arrow keys: orbit camera",)

                shadow_rays = config.cast_shadows or config.is_cornell
                floor_rotation = (
                    orbit_rotation if (config.floor is not None and not config.is_cornell) else config.floor_rotation
                )

                if using_gpu and gpu_display is not None:
                    frame_matrix = cast(
                        FrameMatrix,
                        engine.render(
                            config.mesh,
                            combined_rotation,
                            translation=config.object_translation,
                            floor=config.floor,
                            floor_rotation=floor_rotation,
                            floor_translation=config.floor_translation,
                            cast_shadows=shadow_rays,
                            enable_reflections=config.enable_reflections,
                            hud=hud_lines,
                            hud_color=252,
                            output_format="matrix",
                        ),
                    )
                    gpu_display.draw_matrix(frame_matrix)
                else:
                    frame_str = cast(
                        str,
                        engine.render(
                            config.mesh,
                            combined_rotation,
                            translation=config.object_translation,
                            floor=config.floor,
                            floor_rotation=floor_rotation,
                            floor_translation=config.floor_translation,
                            cast_shadows=shadow_rays,
                            enable_reflections=config.enable_reflections,
                            hud=hud_lines,
                            hud_color=252,
                        ),
                    )
                    terminal = cast(TerminalController, active_display)
                    terminal.draw(frame_str)

                frame_counter += 1
                if args.frames and frame_counter >= args.frames:
                    break

                frame_time = time.perf_counter() - frame_start
                sleep_time = config.frame_duration - frame_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:  # pragma: no cover - interactive loop
            if controller is not None:
                controller.restore()
            sys.stdout.write("\nInterrupted. Bye!\n")
            sys.stdout.flush()


async def _run_async_loop(args: argparse.Namespace, config: RuntimeConfig) -> None:
    controller = config.controller
    display_context = config.display_context
    using_gpu = config.using_gpu
    gpu_display = config.gpu_display

    try:
        with display_context as active_display:
            width, height = active_display.size_tuple()
            engine = _create_engine(width, height, args, config)

            rotation = config.initial_rotation
            rotation_velocity = config.rotation_velocity
            orbit_pitch = 0.0
            orbit_yaw = 0.0
            orbit_step = config.orbit_step
            orbit_pitch_limit = config.orbit_pitch_limit

            frame_counter = 0
            last_frame_start: float | None = None
            smoothed_fps = max(1.0, args.fps)

            executor_workers = max(2, config.async_workers)
            async_chunk_rows = max(config.pixel_step, config.async_chunk_rows)

            with ThreadPoolExecutor(max_workers=executor_workers) as executor:
                while True:
                    frame_start = time.perf_counter()
                    delta = 0.0
                    if last_frame_start is not None:
                        delta = frame_start - last_frame_start
                        instantaneous_fps = 1.0 / max(delta, 1e-6)
                        smoothed_fps = smoothed_fps * 0.85 + instantaneous_fps * 0.15
                    last_frame_start = frame_start

                    if not using_gpu:
                        width, height = active_display.size_tuple()
                        engine.resize(width, height)

                    rotation = _update_rotation(rotation, rotation_velocity, delta)

                    if not config.is_cornell and controller is not None:
                        for key in controller.poll_keys():
                            if key == "LEFT":
                                orbit_yaw += orbit_step
                            elif key == "RIGHT":
                                orbit_yaw -= orbit_step
                            elif key == "UP":
                                orbit_pitch = min(orbit_pitch + orbit_step, orbit_pitch_limit)
                            elif key == "DOWN":
                                orbit_pitch = max(orbit_pitch - orbit_step, -orbit_pitch_limit)

                        if orbit_yaw > math.pi:
                            orbit_yaw -= math.tau
                        elif orbit_yaw < -math.pi:
                            orbit_yaw += math.tau

                    orbit_rotation = Vec3(orbit_pitch, orbit_yaw, 0.0) if not config.is_cornell else Vec3(0.0, 0.0, 0.0)
                    combined_rotation = Vec3(
                        rotation.x + orbit_rotation.x,
                        rotation.y + orbit_rotation.y,
                        rotation.z + orbit_rotation.z,
                    )

                    hud_lines = (f"FPS {smoothed_fps:5.1f}",)
                    if not config.is_cornell:
                        hud_lines = hud_lines + ("Arrow keys: orbit camera",)

                    shadow_rays = config.cast_shadows or config.is_cornell
                    floor_rotation = (
                        orbit_rotation if (config.floor is not None and not config.is_cornell) else config.floor_rotation
                    )

                    if using_gpu and gpu_display is not None:
                        frame_matrix = cast(
                            FrameMatrix,
                            await engine.render_async(
                                config.mesh,
                                combined_rotation,
                                translation=config.object_translation,
                                floor=config.floor,
                                floor_rotation=floor_rotation,
                                floor_translation=config.floor_translation,
                                cast_shadows=shadow_rays,
                                enable_reflections=config.enable_reflections,
                                hud=hud_lines,
                                hud_color=252,
                                output_format="matrix",
                                executor=executor,
                                chunk_rows=async_chunk_rows,
                            ),
                        )
                        gpu_display.draw_matrix(frame_matrix)
                    else:
                        frame_str = cast(
                            str,
                            await engine.render_async(
                                config.mesh,
                                combined_rotation,
                                translation=config.object_translation,
                                floor=config.floor,
                                floor_rotation=floor_rotation,
                                floor_translation=config.floor_translation,
                                cast_shadows=shadow_rays,
                                enable_reflections=config.enable_reflections,
                                hud=hud_lines,
                                hud_color=252,
                                executor=executor,
                                chunk_rows=async_chunk_rows,
                            ),
                        )
                        terminal = cast(TerminalController, active_display)
                        terminal.draw(frame_str)

                    frame_counter += 1
                    if args.frames and frame_counter >= args.frames:
                        break

                    frame_time = time.perf_counter() - frame_start
                    sleep_time = config.frame_duration - frame_time
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
    except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover - interactive loop
        if controller is not None:
            controller.restore()
        sys.stdout.write("\nInterrupted. Bye!\n")
        sys.stdout.flush()


def run() -> None:
    args = parse_arguments()
    config = _setup_runtime(args)
    _emit_warnings(config.warnings)

    if config.async_mode:
        asyncio.run(_run_async_loop(args, config))
    else:
        _run_sync_loop(args, config)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
