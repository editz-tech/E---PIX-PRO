import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from portrait_alignment import (
    _level_shoulders_below_face,
    _normalized_line_angle,
    _rotate_expand,
)


class PortraitAlignmentTests(unittest.TestCase):
    def test_shoulder_line_angle_is_horizontal_direction_independent(self):
        self.assertAlmostEqual(_normalized_line_angle((0, 0), (100, 10)), 5.7106, places=3)
        self.assertAlmostEqual(_normalized_line_angle((100, 10), (0, 0)), 5.7106, places=3)

    def test_rotation_expands_canvas_instead_of_cropping(self):
        image = Image.new("RGBA", (200, 300), (80, 120, 160, 255))
        rotated = _rotate_expand(image, 7.0)
        self.assertGreaterEqual(rotated.width, image.width)
        self.assertGreaterEqual(rotated.height, image.height)
        self.assertEqual(rotated.mode, "RGBA")

    def test_shoulder_warp_changes_lower_body_without_changing_upper_face_area(self):
        class FakePose:
            def process(self, _rgb):
                landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
                landmarks[11] = SimpleNamespace(x=0.25, y=0.40, visibility=1.0)
                landmarks[12] = SimpleNamespace(x=0.75, y=0.48, visibility=1.0)
                return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))

        analyzer = SimpleNamespace(_pose=FakePose())
        pixels = np.zeros((300, 400, 4), dtype=np.uint8)
        for y in range(300):
            pixels[y, :, :3] = y
        pixels[:, :, 3] = 255
        source = Image.fromarray(pixels, "RGBA")

        corrected, measured = _level_shoulders_below_face(
            source, analyzer, (140, 40, 120, 100)
        )
        self.assertIsNotNone(measured)
        self.assertEqual(corrected.size, source.size)
        before = np.asarray(source)
        after = np.asarray(corrected)
        # The upper face area is above the warp transition and must be identical.
        np.testing.assert_array_equal(after[:145], before[:145])
        # The lower image is actually transformed.
        self.assertFalse(np.array_equal(after[190:], before[190:]))


if __name__ == "__main__":
    unittest.main()
