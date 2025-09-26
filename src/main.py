"""Interactive entry point for the terminal 3D renderer demo."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

from .renderer.engine import Mesh, RenderEngine, Vec3
from .renderer.objects import cube_mesh
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
        choices=["cube"],
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
    return parser.parse_args()


def _mesh_factory(name: str, scale: float) -> Mesh:
    factories: dict[str, Callable[[float], Mesh]] = {
        "cube": cube_mesh,
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
    light = _ensure_light_vector(tuple(args.light))

    controller = TerminalController()
    fps = max(1.0, args.fps)
    frame_duration = 1.0 / fps
    mesh = _mesh_factory(args.object, args.scale)

    with controller:
        width, height = controller.size_tuple()
        engine = RenderEngine(
            width,
            height,
            fov_degrees=args.fov,
            camera_distance=args.distance,
            light_direction=light,
        )

        rotation = Vec3(0.0, 0.0, 0.0)
        rotation_velocity = Vec3(0.9, 1.1, 0.6) * args.speed

        frame_counter = 0
        last_time = time.perf_counter()

        try:
            while True:
                now = time.perf_counter()
                delta = now - last_time
                last_time = now

                width, height = controller.size_tuple()
                engine.resize(width, height)

                rotation = Vec3(
                    rotation.x + rotation_velocity.x * delta,
                    rotation.y + rotation_velocity.y * delta,
                    rotation.z + rotation_velocity.z * delta,
                )

                frame = engine.render(mesh, rotation)
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
