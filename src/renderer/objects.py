"""Predefined mesh helpers."""

from __future__ import annotations

from typing import List

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
