"""Terminal-based 3D rendering toolkit."""

from .engine import Mesh, RenderEngine, Triangle, Vec3
from .objects import cornell_box_mesh, cube_mesh, floor_mesh
from .terminal import TerminalController

__all__ = [
    "Mesh",
    "RenderEngine",
    "Triangle",
    "Vec3",
    "cube_mesh",
    "floor_mesh",
    "cornell_box_mesh",
    "TerminalController",
]
