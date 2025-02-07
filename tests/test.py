import os
import cv2
import numpy as np
from target_detection import (
    get_targets_ellipse, PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION,
    target_coordinates_to_sheet_coordinates, get_impacts_coordinates, get_impacts_mask, get_distance
)


TESTS_RESOURCES_PATH = '../tests_ressources'

class ConsoleColors:
    WARNING = '\033[93m'
    SUCCESS = '\033[92m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


def print_message(message: str, color: str):
    print(f'{color}{message}{ConsoleColors.ENDC}')


def get_ellipses_test(folder: str):
    print(f'Running ellipse test for folder {folder}')
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

    if similarity < 0.995:
        raise Exception(f'Ellipses detection failed for folder {folder}, similarity: {similarity:.4f}')
    elif similarity < 0.997:
        print_message(f'Ellipses detection partially succeeded for folder {folder}, similarity: {similarity:.4f}',
                      ConsoleColors.WARNING)
    else:
        print_message(f'Ellipses detection success for folder {folder}, similarity: {similarity:.4f}',
                      ConsoleColors.SUCCESS)


def get_impacts_test(folder: str):
    print(f'Running impacts test for folder {folder}')
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

    if similarity < 0.999:
        raise Exception(f'Impacts mask failed for folder {folder}, similarity: {similarity:.4f}')
    elif similarity < 0.9995:
        print_message(f'Impacts mask partially succeeded for folder {folder}, similarity: {similarity:.4f}',
                      ConsoleColors.WARNING)
    else:
        print_message(f'Impacts mask success for folder {folder}, similarity: {similarity:.4f}', ConsoleColors.SUCCESS)
    split_result = cv2.split(expected_mask)
    hsv_simulate = cv2.merge((split_result[0], split_result[0], split_result[0]))
    bgr_simulate = cv2.cvtColor(hsv_simulate, cv2.COLOR_HSV2BGR)
    real_coordinates = get_impacts_coordinates(bgr_simulate)
    if len(impacts) != len(real_coordinates):
        raise Exception(f'Impacts detection failed for folder {folder}, impacts count: {len(impacts)}')

    print_message(f'Impacts detection success for folder {folder}, impacts count: {len(impacts)}',
                  ConsoleColors.SUCCESS)

    distances = [min(get_distance(real, impact) for impact in impacts) for real in real_coordinates]
    average_distance = np.mean(distances)

    if average_distance > 4:
        raise Exception(f'Impacts coordinates failed for folder {folder}, avg distance: {average_distance:.2f}')
    elif average_distance > 2:
        print_message(
            f'Impacts coordinates partially succeeded for folder {folder}, avg distance: {average_distance:.2f}',
            ConsoleColors.WARNING)
    else:
        print_message(f'Impacts coordinates success for folder {folder}, avg distance: {average_distance:.2f}',
                      ConsoleColors.SUCCESS)


if __name__ == '__main__':
    for folder in os.listdir(TESTS_RESOURCES_PATH):
        folder_path = f'{TESTS_RESOURCES_PATH}/{folder}'
        if folder != 'TODO' and os.path.isdir(folder_path):
            get_ellipses_test(folder)
            get_impacts_test(folder)
