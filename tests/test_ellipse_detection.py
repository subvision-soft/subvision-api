import os
import unittest

import cv2
import numpy as np

from target_detection import (
    get_targets_ellipse, PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION,
    target_coordinates_to_sheet_coordinates
)

TESTS_RESOURCES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), './resources'))


class EllipseDetectionTests(unittest.TestCase):

    def test_ellipses_detection(self):
        for folder in os.listdir(TESTS_RESOURCES_PATH):
            folder_path = f'{TESTS_RESOURCES_PATH}/{folder}'
            if folder != 'TODO' and os.path.isdir(folder_path) and 'WIP' not in folder:
                with self.subTest(folder=folder):
                    self.run_ellipses_test(folder)

    def run_ellipses_test(self, folder: str):
        img_path = f'{TESTS_RESOURCES_PATH}/{folder}/cropped_sheet.jpg'
        expected_mask_path = f'{TESTS_RESOURCES_PATH}/{folder}/expected_visuals.jpg'

        img = cv2.imread(img_path)
        img = cv2.resize(img, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
        ellipses = get_targets_ellipse(img)
        targets_ellipsis = target_coordinates_to_sheet_coordinates(ellipses)

        black_mat = np.zeros((PICTURE_HEIGHT_SHEET_DETECTION, PICTURE_WIDTH_SHEET_DETECTION), np.uint8)
        for value in targets_ellipsis.values():
            cv2.ellipse(black_mat, value, 255, -1)

        expected_mask = cv2.imread(expected_mask_path, cv2.IMREAD_GRAYSCALE)
        expected_mask = cv2.resize(expected_mask, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
        expected_mask = cv2.threshold(expected_mask, 127, 255, cv2.THRESH_BINARY)[1]

        xor_mat = cv2.bitwise_xor(black_mat, expected_mask)
        similarity = 1 - np.count_nonzero(xor_mat) / xor_mat.size

        self.assertGreaterEqual(similarity, 0.995,
                                f'Ellipses detection failed for folder {folder}, similarity: {similarity:.4f}')

if __name__ == '__main__':
    unittest.main()
