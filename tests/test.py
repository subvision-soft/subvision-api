import os
import cv2

from target_detection import get_targets_ellipse


if __name__ == '__main__':
    # iterate on all folders in the test_resources folder
    # ls current directory
    print(os.listdir('../'))

    for folder in os.listdir('../tests_ressources'):


        # if the folder is a directory
        if os.path.isdir(f'../tests_ressources/{folder}'):
            # open cropped sheet with opencv
            img = cv2.imread(f'../tests_ressources/{folder}/cropped_sheet.jpg')
            ellipses = get_targets_ellipse(img)
            print(f'Folder {folder} has {len(ellipses)} targets')

