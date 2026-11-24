import unittest

import numpy as np
import cv2
from PIL import Image

from shadow_matte import build_subject_alpha


class ShadowMatteTests(unittest.TestCase):
    def test_smooth_neutral_shadow_outside_subject_is_removed(self):
        h = w = 120
        raw = np.full((h, w), 0.03, dtype=np.float32)
        cv2.circle(raw, (60, 60), 25, 0.96, -1)
        cv2.circle(raw, (60, 60), 34, 0.46, -1)

        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.circle(rgb, (60, 60), 34, (165, 165, 165), -1)
        cv2.circle(rgb, (60, 60), 25, (35, 35, 35), -1)

        alpha = np.asarray(build_subject_alpha(raw, Image.fromarray(rgb, "RGB")))
        self.assertLess(int(alpha[60, 91]), 20, "neutral grey cast shadow should be removed")
        self.assertGreater(int(alpha[60, 60]), 240, "confident subject core must remain opaque")

    def test_dark_hair_and_saturated_skin_are_not_classified_as_shadow(self):
        h = w = 120
        raw = np.full((h, w), 0.03, dtype=np.float32)
        cv2.circle(raw, (60, 60), 25, 0.96, -1)
        cv2.circle(raw, (60, 60), 34, 0.46, -1)

        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.circle(rgb, (60, 60), 34, (165, 165, 165), -1)
        cv2.circle(rgb, (60, 60), 25, (35, 35, 35), -1)
        rgb[60, 91] = (25, 25, 25)       # dark hair pixel
        rgb[60, 29] = (190, 115, 85)     # saturated skin/ear pixel

        alpha = np.asarray(build_subject_alpha(raw, Image.fromarray(rgb, "RGB")))
        self.assertGreater(int(alpha[60, 91]), 60, "dark hair edge should be retained")
        self.assertGreater(int(alpha[60, 29]), 60, "skin/ear edge should be retained")


if __name__ == "__main__":
    unittest.main()
