"""Predefined mesh helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .engine import Mesh, Triangle, Vec3


def cube_mesh(size: float = 2.0) -> Mesh:
    """Return a colourful cube centred at the origin."""

    half = size / 2.0

    vertices = {
        "lbf": Vec3(-half, -half, -half),  # left-bottom-front
        "rbf": Vec3(half, -half, -half),
        "rtf": Vec3(half, half, -half),
        "ltf": Vec3(-half, half, -half),
        "lbb": Vec3(-half, -half, half),  # left-bottom-back
        "rbb": Vec3(half, -half, half),
        "rtb": Vec3(half, half, half),
        "ltb": Vec3(-half, half, half),
    }

    # Colours correspond to base ANSI cube components (0-5 range)
    colours = {
        "front": (5, 2, 0),
        "back": (0, 5, 2),
        "left": (5, 0, 4),
        "right": (1, 4, 5),
        "top": (5, 5, 0),
        "bottom": (0, 2, 5),
    }

    triangles: List[Triangle] = [
        # Front face (towards camera)
        Triangle((vertices["lbf"], vertices["rtf"], vertices["rbf"]), colours["front"]),
        Triangle((vertices["lbf"], vertices["ltf"], vertices["rtf"]), colours["front"]),
        # Back face
        Triangle((vertices["lbb"], vertices["rbb"], vertices["rtb"]), colours["back"]),
        Triangle((vertices["lbb"], vertices["rtb"], vertices["ltb"]), colours["back"]),
        # Left face
        Triangle((vertices["lbf"], vertices["lbb"], vertices["ltb"]), colours["left"]),
        Triangle((vertices["lbf"], vertices["ltb"], vertices["ltf"]), colours["left"]),
        # Right face
        Triangle((vertices["rbf"], vertices["rtf"], vertices["rtb"]), colours["right"]),
        Triangle((vertices["rbf"], vertices["rtb"], vertices["rbb"]), colours["right"]),
        # Top face
        Triangle((vertices["ltf"], vertices["ltb"], vertices["rtb"]), colours["top"]),
        Triangle((vertices["ltf"], vertices["rtb"], vertices["rtf"]), colours["top"]),
        # Bottom face
        Triangle((vertices["lbf"], vertices["rbf"], vertices["rbb"]), colours["bottom"]),
        Triangle((vertices["lbf"], vertices["rbb"], vertices["lbb"]), colours["bottom"]),
    ]

    return Mesh(triangles)


def floor_mesh(
    size: float = 12.0,
    *,
    tiles: int = 10,
    colours: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((4, 4, 4), (2, 2, 3)),
) -> Mesh:
    """Return a checkerboard floor composed of triangles on the XZ plane."""

    tiles = max(1, tiles)
    half = size / 2.0
    step = size / tiles
    start = -half

    triangles: List[Triangle] = []
    for ix in range(tiles):
        for iz in range(tiles):
            x0 = start + ix * step
            x1 = x0 + step
            z0 = start + iz * step
            z1 = z0 + step

            colour = colours[(ix + iz) % len(colours)] if colours else (2, 2, 2)

            v00 = Vec3(x0, 0.0, z0)
            v10 = Vec3(x1, 0.0, z0)
            v01 = Vec3(x0, 0.0, z1)
            v11 = Vec3(x1, 0.0, z1)

            triangles.append(Triangle((v00, v01, v11), colour))
            triangles.append(Triangle((v00, v11, v10), colour))

    return Mesh(triangles)


def cornell_box_mesh() -> Mesh:
    """Return a Cornell Box scene (walls, light, and two interior blocks)."""

    triangles: List[Triangle] = []

    def quad(v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, colour: Tuple[int, int, int]) -> None:
        triangles.append(Triangle((v0, v1, v2), colour))
        triangles.append(Triangle((v0, v2, v3), colour))

    floor_y = -2.0
    ceiling_y = 2.0
    left_x = -2.0
    right_x = 2.0
    back_z = 4.0
    front_z = 0.0

    white = (4, 4, 4)
    red = (5, 1, 1)
    green = (1, 5, 1)
    light_colour = (5, 5, 3)
    block_colour = (4, 4, 4)

    # Floor
    quad(
        Vec3(left_x, floor_y, front_z),
        Vec3(right_x, floor_y, front_z),
        Vec3(right_x, floor_y, back_z),
        Vec3(left_x, floor_y, back_z),
        white,
    )

    # Ceiling
    quad(
        Vec3(left_x, ceiling_y, front_z),
        Vec3(left_x, ceiling_y, back_z),
        Vec3(right_x, ceiling_y, back_z),
        Vec3(right_x, ceiling_y, front_z),
        white,
    )

    # Back wall
    quad(
        Vec3(left_x, floor_y, back_z),
        Vec3(right_x, floor_y, back_z),
        Vec3(right_x, ceiling_y, back_z),
        Vec3(left_x, ceiling_y, back_z),
        white,
    )

    # Left wall (red)
    quad(
        Vec3(left_x, floor_y, front_z),
        Vec3(left_x, floor_y, back_z),
        Vec3(left_x, ceiling_y, back_z),
        Vec3(left_x, ceiling_y, front_z),
        red,
    )

    # Right wall (green)
    quad(
        Vec3(right_x, floor_y, back_z),
        Vec3(right_x, floor_y, front_z),
        Vec3(right_x, ceiling_y, front_z),
        Vec3(right_x, ceiling_y, back_z),
        green,
    )

    # Ceiling light patch
    light_scale = 0.6
    light_x0 = -light_scale
    light_x1 = light_scale
    light_z0 = 1.6
    light_z1 = 2.6
    quad(
        Vec3(light_x0, ceiling_y - 1e-3, light_z0),
        Vec3(light_x1, ceiling_y - 1e-3, light_z0),
        Vec3(light_x1, ceiling_y - 1e-3, light_z1),
        Vec3(light_x0, ceiling_y - 1e-3, light_z1),
        light_colour,
    )

    def box(origin: Vec3, size: Vec3, colour: Tuple[int, int, int]) -> None:
        ox, oy, oz = origin.x, origin.y, origin.z
        sx, sy, sz = size.x, size.y, size.z
        corners = {
            "lbf": Vec3(ox, oy, oz),
            "rbf": Vec3(ox + sx, oy, oz),
            "rtf": Vec3(ox + sx, oy + sy, oz),
            "ltf": Vec3(ox, oy + sy, oz),
            "lbb": Vec3(ox, oy, oz + sz),
            "rbb": Vec3(ox + sx, oy, oz + sz),
            "rtb": Vec3(ox + sx, oy + sy, oz + sz),
            "ltb": Vec3(ox, oy + sy, oz + sz),
        }

        faces: Sequence[Tuple[str, str, str, str]] = (
            ("lbf", "rbf", "rtf", "ltf"),  # front
            ("rbb", "lbb", "ltb", "rtb"),  # back
            ("lbb", "lbf", "ltf", "ltb"),  # left
            ("rbf", "rbb", "rtb", "rtf"),  # right
            ("ltf", "rtf", "rtb", "ltb"),  # top
            ("lbb", "rbb", "rbf", "lbf"),  # bottom
        )
        for v0, v1, v2, v3 in faces:
            quad(corners[v0], corners[v1], corners[v2], corners[v3], colour)

    # Short box (left side)
    box(
        origin=Vec3(-1.3, floor_y, 2.2),
        size=Vec3(1.1, 1.4, 1.1),
        colour=block_colour,
    )

    # Tall box (right side)
    box(
        origin=Vec3(0.4, floor_y, 1.1),
        size=Vec3(1.0, 2.5, 1.0),
        colour=block_colour,
    )

    return Mesh(triangles)
