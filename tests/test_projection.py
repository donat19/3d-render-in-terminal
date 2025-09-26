import math
import unittest

from src.renderer.engine import RenderEngine, Vec3


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


if __name__ == "__main__":
    unittest.main()
