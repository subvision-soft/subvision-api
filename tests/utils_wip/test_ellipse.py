import os
import cv2
import numpy as np
import unittest
from target_detection import (
    get_targets_ellipse, PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION,
    target_coordinates_to_sheet_coordinates, get_impacts_coordinates, get_impacts_mask, get_distance
)

TESTS_RESOURCES_PATH = '../../tests_ressources'


class TargetDetectionTests(unittest.TestCase):

    def test_ellipses_detection(self):
        for folder in os.listdir(TESTS_RESOURCES_PATH):
            folder_path = f'{TESTS_RESOURCES_PATH}/{folder}'
            if folder != 'TODO' and os.path.isdir(folder_path):
                with self.subTest(folder=folder):
                    self.run_ellipses_test(folder)

    def test_impacts_detection(self):
        for folder in os.listdir(TESTS_RESOURCES_PATH):
            folder_path = f'{TESTS_RESOURCES_PATH}/{folder}'
            if folder != 'TODO' and os.path.isdir(folder_path):
                with self.subTest(folder=folder):
                    self.run_impacts_test(folder)

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

    def run_impacts_test(self, folder: str):
        img_path = f'{TESTS_RESOURCES_PATH}/{folder}/cropped_sheet.jpg'
        mask_path = f'{TESTS_RESOURCES_PATH}/{folder}/mask_impacts.jpg'

        img = cv2.imread(img_path)
        img = cv2.resize(img, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
        impacts = get_impacts_coordinates(img)

        expected_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        expected_mask = cv2.resize(expected_mask, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
        expected_mask = cv2.threshold(expected_mask, 127, 255, cv2.THRESH_BINARY)[1]

        mask_impacts = get_impacts_mask(img)
        xor_mat = cv2.bitwise_xor(mask_impacts, expected_mask)
        similarity = 1 - np.count_nonzero(xor_mat) / xor_mat.size

        self.assertGreaterEqual(similarity, 0.999,
                                f'Impacts mask failed for folder {folder}, similarity: {similarity:.4f}')

        split_result = cv2.split(expected_mask)
        hsv_simulate = cv2.merge((split_result[0], split_result[0], split_result[0]))
        bgr_simulate = cv2.cvtColor(hsv_simulate, cv2.COLOR_HSV2BGR)
        real_coordinates = get_impacts_coordinates(bgr_simulate)
        self.assertEqual(len(impacts), len(real_coordinates),
                         f'Impacts detection failed for folder {folder}, impacts count: {len(impacts)}')

        distances = [min(get_distance(real, impact) for impact in impacts) for real in real_coordinates]
        average_distance = np.mean(distances)
        self.assertLessEqual(average_distance, 4,
                             f'Impacts coordinates failed for folder {folder}, avg distance: {average_distance:.2f}')


if __name__ == '__main__':
    unittest.main()
