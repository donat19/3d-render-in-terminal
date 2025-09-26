import unittest

from src.renderer.engine import RenderEngine, Vec3
from src.renderer.objects import cornell_box_mesh, cube_mesh, floor_mesh


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RenderEngine(120, 60, fov_degrees=90.0, camera_distance=4.0)

    def test_projection_centred(self) -> None:
        projected = self.engine.project_point(Vec3(0.0, 0.0, 4.0))
        self.assertIsNotNone(projected)
        x, y, _ = projected  # type: ignore[assignment]
        self.assertAlmostEqual(x, (self.engine.width - 1) / 2, places=5)
        self.assertAlmostEqual(y, (self.engine.height - 1) / 2, places=5)

    def test_projection_ordering(self) -> None:
        left = self.engine.project_point(Vec3(-1.0, 0.0, 4.0))
        right = self.engine.project_point(Vec3(1.0, 0.0, 4.0))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertLess(left[0], right[0])  # type: ignore[index]

    def test_vec3_normalisation_safe(self) -> None:
        zero = Vec3(0.0, 0.0, 0.0).normalized()
        self.assertEqual(zero, Vec3(0.0, 0.0, 0.0))

        unit = Vec3(2.0, 0.0, 0.0).normalized()
        self.assertAlmostEqual(unit.length(), 1.0)

    def test_floor_mesh_tile_count(self) -> None:
        mesh = floor_mesh(size=8.0, tiles=4)
        self.assertEqual(len(mesh.triangles), 4 * 4 * 2)

    def test_shadow_renders_on_floor(self) -> None:
        engine = RenderEngine(120, 60, fov_degrees=70.0, camera_distance=6.0)
        cube = cube_mesh(2.0)
        floor = floor_mesh(size=10.0, tiles=6)
        frame_without_shadows = engine.render(
            cube,
            rotation=Vec3(0.3, 0.9, 0.1),
            translation=Vec3(0.0, 0.0, 0.0),
            floor=floor,
            floor_rotation=Vec3(0.0, 0.0, 0.0),
            floor_translation=Vec3(0.0, -2.0, 0.0),
            cast_shadows=False,
            enable_reflections=False,
        )

        frame_with_shadows = engine.render(
            cube,
            rotation=Vec3(0.3, 0.9, 0.1),
            translation=Vec3(0.0, 0.0, 0.0),
            floor=floor,
            floor_rotation=Vec3(0.0, 0.0, 0.0),
            floor_translation=Vec3(0.0, -2.0, 0.0),
            cast_shadows=True,
            enable_reflections=False,
        )

        self.assertNotEqual(frame_without_shadows, frame_with_shadows)

    def test_hud_overlay_displays_text(self) -> None:
        engine = RenderEngine(60, 30, fov_degrees=70.0, camera_distance=6.0)
        cube = cube_mesh(1.5)
        frame = engine.render(
            cube,
            rotation=Vec3(0.0, 0.0, 0.0),
            hud=("FPS 60.0",),
            hud_color=None,
        )
        first_line = frame.splitlines()[0]
        self.assertIn("FPS 60.0", first_line)

    def test_cornell_box_mesh_not_empty(self) -> None:
        mesh = cornell_box_mesh()
        self.assertGreater(len(mesh.triangles), 0)


if __name__ == "__main__":
    unittest.main()
