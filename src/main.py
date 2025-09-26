"""Interactive entry point for the terminal 3D renderer demo."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Callable

from .renderer.engine import Mesh, RenderEngine, Vec3
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


def run() -> None:
    args = parse_arguments()
    is_cornell = args.object == "cornell"

    default_light = (-0.4, 0.8, -0.6)
    requested_light = tuple(args.light)
    if is_cornell and requested_light == default_light:
        requested_light = (0.2, -1.0, -0.2)
    light = _ensure_light_vector(requested_light)

    controller = TerminalController()
    fps = max(1.0, args.fps)
    frame_duration = 1.0 / fps
    mesh = _mesh_factory(args.object, args.scale)

    floor = None
    floor_translation = Vec3(0.0, -0.5 * args.scale - 1.1, 0.0)
    cast_shadows = False
    enable_reflections = False
    object_translation = Vec3(0.0, 0.0, 0.0)
    floor_rotation = Vec3(0.0, 0.0, 0.0)

    if not is_cornell:
        floor = None if args.no_floor else floor_mesh(args.floor_size, tiles=max(1, args.floor_tiles))
        cast_shadows = floor is not None and not args.no_shadows
        enable_reflections = floor is not None and not args.no_reflections
        object_translation = Vec3(0.0, 0.0, 0.0)
    else:
        # Position the Cornell box so the camera peers inside without rotation.
        object_translation = Vec3(0.0, -0.25, -1.0)

    camera_distance = args.distance
    if is_cornell and camera_distance < 6.0:
        camera_distance = 6.0

    with controller:
        width, height = controller.size_tuple()
        engine = RenderEngine(
            width,
            height,
            fov_degrees=args.fov,
            camera_distance=camera_distance,
            light_direction=light,
        )

        if is_cornell and args.fov == 70.0:
            engine.set_fov(50.0)

        rotation = Vec3(0.0, 0.0, 0.0)
        if is_cornell:
            rotation = Vec3(0.1, 0.3, 0.0)
        if is_cornell:
            rotation_velocity = Vec3(0.0, 0.0, 0.0)
        else:
            rotation_velocity = Vec3(0.9, 1.1, 0.6) * args.speed

        orbit_pitch = 0.0
        orbit_yaw = 0.0
        orbit_step = math.radians(3.0)
        orbit_pitch_limit = math.radians(65.0)

        frame_counter = 0
        last_time = time.perf_counter()
        smoothed_fps = max(1.0, args.fps)

        try:
            while True:
                now = time.perf_counter()
                delta = now - last_time
                last_time = now
                instantaneous_fps = 1.0 / max(delta, 1e-6)
                smoothed_fps = smoothed_fps * 0.85 + instantaneous_fps * 0.15

                width, height = controller.size_tuple()
                engine.resize(width, height)

                if rotation_velocity.length_squared() > 0.0:
                    rotation = Vec3(
                        rotation.x + rotation_velocity.x * delta,
                        rotation.y + rotation_velocity.y * delta,
                        rotation.z + rotation_velocity.z * delta,
                    )

                if not is_cornell:
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

                orbit_rotation = Vec3(orbit_pitch, orbit_yaw, 0.0) if not is_cornell else Vec3(0.0, 0.0, 0.0)
                combined_rotation = Vec3(
                    rotation.x + orbit_rotation.x,
                    rotation.y + orbit_rotation.y,
                    rotation.z + orbit_rotation.z,
                )

                hud_lines = (f"FPS {smoothed_fps:5.1f}",)
                if not is_cornell:
                    hud_lines = hud_lines + ("Arrow keys: orbit camera",)

                shadow_rays = cast_shadows or is_cornell

                frame = engine.render(
                    mesh,
                    combined_rotation,
                    translation=object_translation,
                    floor=floor,
                    floor_rotation=orbit_rotation if (floor is not None and not is_cornell) else floor_rotation,
                    floor_translation=floor_translation,
                    cast_shadows=shadow_rays,
                    enable_reflections=enable_reflections,
                    hud=hud_lines,
                    hud_color=252,
                )
                controller.draw(frame)

                frame_counter += 1
                if args.frames and frame_counter >= args.frames:
                    break

                elapsed = time.perf_counter() - now
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            controller.restore()
            sys.stdout.write("\nInterrupted. Bye!\n")
            sys.stdout.flush()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
