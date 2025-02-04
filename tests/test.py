import json
import os
import cv2
import numpy as np

from target_detection import get_targets_ellipse, PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION, \
    target_coordinates_to_sheet_coordinates, get_impacts_coordinates, get_impacts_mask, get_distance


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def warning_print(message:str):
    print(f'{bcolors.WARNING}{message}{bcolors.ENDC}')

def success_print(message:str):
    print(f'{bcolors.OKGREEN}{message}{bcolors.ENDC}')

def error_print(message:str):
    print(f'{bcolors.FAIL}{message}{bcolors.ENDC}')


def get_ellipses_test(folder:str):
    print(f'Running test for folder {folder}')
    # open cropped sheet with opencv
    img = cv2.imread(f'../tests_ressources/{folder}/cropped_sheet.jpg')
    img = cv2.resize(img, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
    ellipses = get_targets_ellipse(img)
    targets_ellipsis = target_coordinates_to_sheet_coordinates(ellipses)
    # create a black image
    black_mat = np.zeros((PICTURE_HEIGHT_SHEET_DETECTION, PICTURE_WIDTH_SHEET_DETECTION, 3), np.uint8)
    for key, value in targets_ellipsis.items():
        cv2.ellipse(black_mat, value, (255, 255, 255), -1)
    expected_mask = cv2.imread(f'../tests_ressources/{folder}/expected_visuals.jpg')
    expected_mask = cv2.resize(expected_mask, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
    expected_mask = cv2.cvtColor(expected_mask, cv2.COLOR_BGR2GRAY)
    expected_mask = cv2.threshold(expected_mask, 127, 255, cv2.THRESH_BINARY)[1]
    black_mat = cv2.cvtColor(black_mat, cv2.COLOR_BGR2GRAY)
    xor_mat = cv2.bitwise_xor(black_mat, expected_mask)
    # get percentage of similarity
    similarity = 1 - np.count_nonzero(xor_mat) / xor_mat.size
    if similarity < 0.995:
        raise Exception(f'Ellipses detections failed for folder {folder}, similarity: {similarity}')
    elif similarity < 0.997:
        warning_print(f'Ellipses detections partially success for folder {folder}, similarity: {similarity}')

    success_print(f'Ellipses detections success for folder {folder}, similarity: {similarity}')


def get_impacts_test(folder):
    img = cv2.imread(f'../tests_ressources/{folder}/cropped_sheet.jpg')
    img = cv2.resize(img, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
    impacts = get_impacts_coordinates(img)
    with open(f'../tests_ressources/{folder}/data.json', 'r') as file:
        data_json = json.load(file)
        # get attributes from json
        impacts_count = data_json['impactsCount']
        if len(impacts) != impacts_count:
            raise Exception(f'Impacts detections failed for folder {folder}, impacts count: {len(impacts)}')
        success_print(f'Impacts detections success for folder {folder}, impacts count: {len(impacts)}')
    expected_mask_impacts = cv2.imread(f'../tests_ressources/{folder}/mask_impacts.jpg')
    expected_mask_impacts = cv2.resize(expected_mask_impacts, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
    expected_mask_impacts = cv2.cvtColor(expected_mask_impacts, cv2.COLOR_BGR2GRAY)
    expected_mask_impacts = cv2.threshold(expected_mask_impacts, 127, 255, cv2.THRESH_BINARY)[1]
    mask_impacts = get_impacts_mask(img)

    xor_mat = cv2.bitwise_xor(mask_impacts, expected_mask_impacts)
    # get percentage of similarity
    similarity = 1 - np.count_nonzero(xor_mat) / xor_mat.size
    if similarity < 0.999:
        raise Exception(f'Impacts mask failed for folder {folder}, similarity: {similarity}')
    elif similarity < 0.9995:
        warning_print(f'Impacts mask partially success for folder {folder}, similarity: {similarity}')
    success_print(f'Impacts mask success for folder {folder}, similarity: {similarity}')
    expected_mask_impacts = cv2.imread(f'../tests_ressources/{folder}/mask_impacts.jpg')
    expected_mask_impacts = cv2.resize(expected_mask_impacts, (PICTURE_WIDTH_SHEET_DETECTION, PICTURE_HEIGHT_SHEET_DETECTION))
    split_result = cv2.split(expected_mask_impacts)
    hsv_simulate = cv2.merge((split_result[0], split_result[0], split_result[0]))
    bgr_simulate = cv2.cvtColor(hsv_simulate, cv2.COLOR_HSV2BGR)
    real_coordinates = get_impacts_coordinates(bgr_simulate)
    # get average distance between real and detected impacts
    distances = []
    for real_impact in real_coordinates:
        min_distance = 100000
        closest_impact = None
        for impact in impacts:
            distance = get_distance(real_impact, impact)
            if distance < min_distance:
                min_distance = distance
                closest_impact = impact

        cv2.line(expected_mask_impacts, (int(real_impact[0]), int(real_impact[1])), (int(closest_impact[0]), int(closest_impact[1])), (0, 0, 255), 2)
        distances.append(min_distance)

    average_distance = np.mean(distances)
    # convert to mm (

    if average_distance > 4:
        raise Exception(f'Impacts coordinates failed for folder {folder}, average distance: {average_distance}')
    elif average_distance > 2:
        warning_print(f'Impacts coordinates partially success for folder {folder}, average distance: {average_distance}')
    success_print(f'Impacts coordinates success for folder {folder}, average distance: {average_distance}')




if __name__ == '__main__':
    # iterate on all folders in the test_resources folder
    # ls current directory
    print(os.listdir('../'))
    for folder in os.listdir('../tests_ressources'):
        if folder == 'TODO':
            continue
        # if the folder is a directory
        if os.path.isdir(f'../tests_ressources/{folder}'):
            get_ellipses_test(folder)
            get_impacts_test(folder)




