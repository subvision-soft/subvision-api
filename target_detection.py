import math
from enum import Enum
import timeit

import cv2
import numpy as np
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from numpy import ndarray
class Zone(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    CENTER = 5
PICTURE_SIZE_SHEET_DETECTION = 1000
model = YOLO("nano_semantic_model.pt")


def to_radians(angle):
    return angle * math.pi / 180

def rotate_point(center, point, angle):
    s = math.sin(angle)
    c = math.cos(angle)

    # Translate point back to origin
    translated_x = point[0] - center[0]
    translated_y = point[1] - center[1]

    # Rotate point
    rotated_x = translated_x * c - translated_y * s
    rotated_y = translated_x * s + translated_y * c

    # Translate point back
    new_point = (rotated_x + center[0], rotated_y + center[1])

    return new_point


def get_point_on_ellipse(ellipse, angle):
    center, radii,ellipse_angle = ellipse
    radius_x, radius_y = radii
    x = center[0] + math.cos(angle) * (radius_x /2)
    y = center[1] + math.sin(angle) * (radius_y /2)
    rotated_point = rotate_point(center, (x, y),to_radians(ellipse_angle) + to_radians(180))
    # Convert the coordinates to integers
    return (int(x), int(y))

def grow_ellipse(ellipse, factor):
    center, radii, angle = ellipse
    return (center, (radii[0] * factor, radii[1] * factor), angle)


def get_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_angle(point1, point2):
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])

def get_real_distance(center, border, impact):
    length = get_distance(center, border)
    distance = get_distance(center, impact)
    percent = distance / length
    real_length = 25
    millimeter_distance = real_length * percent
    return round(millimeter_distance)

def get_points(distance):
    point = 570
    i = 0
    maximum_impact_distance = 48
    if distance > maximum_impact_distance:
        return 0
    for i in range(5):
        if distance <= 0:
            break
        point -= 6
        distance -= 1
    for i in range(i, 48):
        if distance <= 0:
            break
        point -= 3
        distance -= 1
    return point

def draw_cross_at_coordinates(image: np.ndarray, coordinates: list, color: tuple = (0, 255, 0), size: int = 20, width: int = 2):
    for coordinate in coordinates:
        # Convert coordinates to integers in case they are floats
        x, y = int(coordinate[0]), int(coordinate[1])

        # Draw horizontal line of the cross
        # check if the line is in the image
        if x - size >= 0 and x + size < image.shape[1] and y >= 0 and y < image.shape[0]:
            cv2.line(image, (x - size, y), (x + size, y), color, width)
            # Draw vertical line of the cross
            cv2.line(image, (x, y - size), (x, y + size), color, width)
def enhance_image_for_edge_detection(image: ndarray, blur_radius: int = 0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, saturation, v = cv2.split(hsv)
    if blur_radius > 0:
        v = cv2.blur(v, (blur_radius, blur_radius))
    return v


def get_edges(image: ndarray, blur_radius: int = 0, canny_threshold_1: int = 100, canny_threshold_2: int = 200):
    image_clone = image.copy()
    if blur_radius > 0:
        image_clone = cv2.blur(image_clone, (blur_radius, blur_radius))
    return cv2.Canny(image_clone, canny_threshold_1, canny_threshold_2)

def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(value, max_value))


def get_biggest_valid_contour(contours):
    biggest_contour = None
    biggest_area = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.01, True)
        if len(approx) != 4 or cv2.contourArea(approx) < biggest_area:
            continue
        angles = [math.acos(clamp(
            ((p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1])) /
            (((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 *
             ((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2) ** 0.5))) * 180 / math.pi
                  for i in range(4)
                  for p1, p2, p3 in [(approx[i][0], approx[(i + 1) % 4][0], approx[(i + 2) % 4][0])]]
        if any(angle < 70 or angle > 110 for angle in angles):
            continue
        area = cv2.contourArea(approx)
        if area / (PICTURE_SIZE_SHEET_DETECTION * PICTURE_SIZE_SHEET_DETECTION) < 0.1 or area / (
                PICTURE_SIZE_SHEET_DETECTION * PICTURE_SIZE_SHEET_DETECTION) > 0.9:
            continue
        biggest_contour = approx
        biggest_area = area
    return biggest_contour


def coordinates_to_percentage(coordinates, width, height):
    percentage_coordinates = []
    for coordinate in coordinates:
        percentage_coordinates.append((coordinate[0] / width, coordinate[1] / height))
    return percentage_coordinates


def get_sheet_coordinates(sheet_mat: ndarray):
    mat_resized = cv2.resize(sheet_mat.copy(), (PICTURE_SIZE_SHEET_DETECTION, PICTURE_SIZE_SHEET_DETECTION))
    results = model.track(mat_resized, verbose=False, conf=0.5)
    if results[0].masks is not None:
        #get index of best detection
        best_detection_index =np.argsort(results[0].boxes.conf)[-1:]
        mask = results[0].masks.data.cpu().numpy()[best_detection_index]
        # Convert mask to 8-bit single-channel image
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (PICTURE_SIZE_SHEET_DETECTION, PICTURE_SIZE_SHEET_DETECTION))
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = get_biggest_valid_contour(contours)
        #draw the contour
        if biggest_contour is None:
            return None

        # Get coordinates of the biggest contour and convert to percentage
        return coordinates_to_percentage(
            [(biggest_contour[i][0][0], biggest_contour[i][0][1]) for i in range(4)],
            PICTURE_SIZE_SHEET_DETECTION,
            PICTURE_SIZE_SHEET_DETECTION
        )
    return None
def extract_sheet(image):
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 3)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img










def draw_sheet_detection_contour(image, coordinates):
    coordinates_len = len(coordinates)
    for i in range(coordinates_len):
        cv2.line(image, coordinates[i], coordinates[(i + 1) % coordinates_len], (0, 255, 0), 3)


def percentage_to_coordinates(percentage_coordinates, width, height):
    coordinates = []
    for percentage_coordinate in percentage_coordinates:
        coordinates.append((int(percentage_coordinate[0] * width), int(percentage_coordinate[1] * height)))
    return coordinates

def get_sheet_picture(image: ndarray):
    coordinates = get_sheet_coordinates(image)
    if coordinates is None:
        return None
    height, width, _ = image.shape
    real_coordinates = percentage_to_coordinates(coordinates, width, height)
    approx = np.array(real_coordinates, np.float32)
    target_coordinates = np.array([
        [0, 0],
        [PICTURE_SIZE_SHEET_DETECTION, 0],
        [PICTURE_SIZE_SHEET_DETECTION, PICTURE_SIZE_SHEET_DETECTION], [0, PICTURE_SIZE_SHEET_DETECTION]
    ], np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(approx, target_coordinates)
    return cv2.warpPerspective(image, transformation_matrix,
                               (PICTURE_SIZE_SHEET_DETECTION, PICTURE_SIZE_SHEET_DETECTION))

def get_crop_coordinates(image: ndarray, target_zone: Zone):
    height, width, _ = image.shape
    x1, x2, y1, y2 = 0, width, 0, height

    if target_zone in [Zone.TOP_RIGHT, Zone.BOTTOM_RIGHT]:
        x1 = int(width / 2)
    if target_zone in [Zone.BOTTOM_LEFT, Zone.BOTTOM_RIGHT]:
        y1 = int(height / 2)
    if target_zone in [Zone.TOP_LEFT, Zone.BOTTOM_LEFT]:
        x2 = int(width / 2)
    if target_zone in [Zone.TOP_LEFT, Zone.TOP_RIGHT]:
        y2 = int(height / 2)
    if target_zone == Zone.CENTER:
        x1, y1 = int(width / 4), int(height / 4)
        x2, y2 = width - x1, height - y1

    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}


def get_target_picture(sheet_mat: ndarray, target_zone: Zone):
    sheet_mat_clone = sheet_mat.copy()
    coordinates = get_crop_coordinates(sheet_mat_clone, target_zone)
    return sheet_mat_clone[coordinates['x1']:coordinates['x2'], coordinates['y1']:coordinates['y2']]


def get_biggest_contour(contours):
    max_area = 0
    biggest_contour = None
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        area = bounding_rect[2] * bounding_rect[3]
        if area > max_area:
            max_area = area
            biggest_contour = contour
    return biggest_contour


def get_impacts_mask(image: ndarray):

    image = cv2.blur(image.copy(), (PICTURE_SIZE_SHEET_DETECTION // 100, PICTURE_SIZE_SHEET_DETECTION // 100))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, saturation, v = cv2.split(hsv)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saturation)
    min_val = (max_val - min_val) / 1.4 + min_val

    lower_bound = np.array([min_val], dtype=np.uint8)
    upper_bound = np.array([max_val], dtype=np.uint8)

    low = np.full((saturation.shape[0], saturation.shape[1]), lower_bound, dtype=np.uint8)
    high = np.full((saturation.shape[0], saturation.shape[1]), upper_bound, dtype=np.uint8)

    cv2.inRange(saturation, low, high, saturation)

    contours, hierarchy = cv2.findContours(saturation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # create a mask image that contains the contour filled in
    mask = np.zeros_like(saturation)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, -1, 255, -1)  # Draw filled contour in mask
    return mask

def get_impacts_coordinates(image: ndarray) -> list:
    mask = get_impacts_mask(image)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if len(contour) >= 5]
    ellipses = [cv2.fitEllipse(contour) for contour in contours]
    # get ellipses centers
    # remove ellipses with nan values
    ellipses = [ellipse for ellipse in ellipses if not math.isnan(ellipse[0][0]) and not math.isnan(ellipse[0][1])]
    centers = [(int(ellipse[0][0]), int(ellipse[0][1])) for ellipse in ellipses]
    return centers




def get_color_mask(mat: ndarray, color: tuple):
    color_mat = np.full((1, 1, 3), color, dtype=np.uint8)
    hsv = cv2.cvtColor(color_mat, cv2.COLOR_RGB2HSV)
    min_val = np.array([hsv[0][0][0] - 10, 100, 50], dtype=np.uint8)
    max_val = np.array([hsv[0][0][0] + 10, 255, 255], dtype=np.uint8)
    hsv_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2HSV)
    min_mat = np.full((hsv_mat.shape[0], hsv_mat.shape[1], 3), min_val, dtype=np.uint8)
    high_mat = np.full((hsv_mat.shape[0], hsv_mat.shape[1], 3), max_val, dtype=np.uint8)
    mask = cv2.inRange(hsv_mat, min_mat, high_mat)
    kernel = np.ones((PICTURE_SIZE_SHEET_DETECTION // 200, PICTURE_SIZE_SHEET_DETECTION // 200), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask
def get_target_ellipse(mat,zone:Zone):
    circle = np.zeros((mat.shape[1], mat.shape[0]), dtype=np.uint8)
    cv2.circle(circle, (mat.shape[1] // 2, mat.shape[0] // 2), int(mat.shape[1] / 2.2), (255, 255, 255), -1)

    kernel_size = PICTURE_SIZE_SHEET_DETECTION // 200
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    hsv = cv2.cvtColor(mat.copy(), cv2.COLOR_BGR2HSV)
    hsv_channels = cv2.split(hsv)
    value = hsv_channels[2]

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(value)
    max_val = (max_val - min_val) / 2 + min_val

    min_mat = np.full((value.shape[0], value.shape[1]), min_val, dtype=value.dtype)
    high_mat = np.full((value.shape[0], value.shape[1]), max_val, dtype=value.dtype)
    value_mask = cv2.inRange(value, min_mat, high_mat)
    cv2.bitwise_and(value_mask, circle, value_mask)

    impacts = get_impacts_mask(mat)
    # remove impacts from the mask
    cv2.bitwise_and(value_mask, cv2.bitwise_not(impacts), value_mask)

    close = cv2.morphologyEx(value_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    close = cv2.morphologyEx(close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(biggest_contour)
    for i in range(10):
        empty = np.zeros(mat.shape, dtype=np.uint8)
        cv2.ellipse(empty, ellipse, (255, 255, 255), -1)
        empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
        xor = cv2.bitwise_xor(empty_gray, close)
        cv2.bitwise_or(close, xor, xor)
        contours, _ = cv2.findContours(xor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(biggest_contour)
        if ellipse[1][0] < ellipse[1][1] * 0.7 or ellipse[1][0] > ellipse[1][1] * 1.3:
            raise ValueError('Problem during visual detection')
    return ellipse


def get_targets_ellipse(image: ndarray):
    zones = [Zone.TOP_LEFT, Zone.TOP_RIGHT, Zone.CENTER, Zone.BOTTOM_LEFT, Zone.BOTTOM_RIGHT]
    ellipsis = {}
    for zone in zones:
        target_mat = get_target_picture(image, zone)
        ellipsis[zone] = get_target_ellipse(target_mat,zone)
    return ellipsis

def target_coordinates_to_sheet_coordinates(ellipsis: dict):
    new_ellipsis = {}
    for key, value in ellipsis.items():

        if key == Zone.TOP_LEFT:
            new_ellipsis[key] = (value[0], value[1], value[2])
        elif key == Zone.TOP_RIGHT:
            new_ellipsis[key] = ((value[0][0], value[0][1] + PICTURE_SIZE_SHEET_DETECTION // 2), value[1], value[2])
        elif key == Zone.BOTTOM_LEFT:
            new_ellipsis[key] = ((value[0][0] + PICTURE_SIZE_SHEET_DETECTION // 2, value[0][1]), value[1], value[2])
        elif key == Zone.BOTTOM_RIGHT:
            new_ellipsis[key] = (
                (value[0][0] + PICTURE_SIZE_SHEET_DETECTION // 2, value[0][1] + PICTURE_SIZE_SHEET_DETECTION // 2),
                value[1], value[2])
        elif key == Zone.CENTER:
            new_ellipsis[key] = (
                (value[0][0] + PICTURE_SIZE_SHEET_DETECTION // 4, value[0][1] + PICTURE_SIZE_SHEET_DETECTION // 4),
                value[1], value[2])
    return new_ellipsis



def process_image(image: ndarray):
    sheet_mat = get_sheet_picture(image)
    if sheet_mat is None:
        return None, None
    coordinates = get_targets_ellipse(sheet_mat)
    coordinates = target_coordinates_to_sheet_coordinates(coordinates)
    impacts = get_impacts_coordinates(sheet_mat)

    #iterate key and value
    for key, ellipse_contrat in coordinates.items():

        drawing_width = 2

        # Draw the ellipse
        target_color = (0, 255, 0)
        # Grow the ellipse
        ellipse_cross_tip = grow_ellipse(ellipse_contrat, 2.2)
        ellipse_mouche = grow_ellipse(ellipse_contrat, 0.2)
        ellipse_petit_blanc = grow_ellipse(ellipse_contrat, 0.6)
        ellipse_moyen_blanc = grow_ellipse(ellipse_contrat, 1.4)
        ellipse_grand_blanc = grow_ellipse(ellipse_contrat, 1.8)
        # Draw the ellipses
        cv2.ellipse(sheet_mat, ellipse_contrat, target_color, drawing_width)
        cv2.ellipse(sheet_mat, ellipse_mouche, target_color, drawing_width)
        cv2.ellipse(sheet_mat, ellipse_petit_blanc, target_color, drawing_width)
        cv2.ellipse(sheet_mat, ellipse_moyen_blanc, target_color, drawing_width)
        cv2.ellipse(sheet_mat, ellipse_grand_blanc, target_color, drawing_width)



        top_point = get_point_on_ellipse(ellipse_cross_tip, to_radians(90))
        bottom_point = get_point_on_ellipse(ellipse_cross_tip, to_radians(270))
        left_point = get_point_on_ellipse(ellipse_cross_tip, to_radians(180))
        right_point = get_point_on_ellipse(ellipse_cross_tip, to_radians(0))
        cv2.line(sheet_mat, top_point, bottom_point, target_color, drawing_width)
        cv2.line(sheet_mat, left_point, right_point, target_color, drawing_width)

    draw_cross_at_coordinates(sheet_mat, impacts,color=(0,165,255),width=3)

    for impact in impacts:
        min_distance = float('inf')
        closest_zone = None

        for key,coordinate in coordinates.items():
            distance = get_distance(impact, coordinate[0])
            if min_distance > distance:
                min_distance = distance
                closest_zone = key

        if closest_zone is None:
            raise ValueError('Aucune zone trouvée')

        rad_angle = get_angle(impact, coordinates[closest_zone][0])
        angle = rad_angle + to_radians(360)
        ellipse_angle = to_radians(coordinates[closest_zone][2] + 360)
        angle = angle - ellipse_angle
        point_on_ellipse = get_point_on_ellipse(
            coordinates[closest_zone],
            angle
        )
        point_on_ellipse = rotate_point(
            coordinates[closest_zone][0],
            point_on_ellipse,
            ellipse_angle + to_radians(180)
        )

        real_distance = get_real_distance(
            coordinates[closest_zone][0],
            point_on_ellipse,
            impact
        )
        points = get_points(real_distance)

        cv2.putText(
            sheet_mat,
            str(points),
            impact,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [0, 0, 0, 255],
            6
        )
        cv2.putText(
            sheet_mat,
            str(points),
            impact,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [255, 255, 255, 255],
            3
        )
    return coordinates
#
# image = cv2.imread('images/sheet.jpg')
# numruns = 10
# duration = timeit.Timer('process_image(image)', globals=globals()).timeit(numruns)
# print(f"Duration: {duration / numruns:.2f} seconds")
# sheet_mat, coordinates = process_image(image)
# if sheet_mat is not None:
#     plt.imshow(sheet_mat)
#     plt.show()
# cv2.waitKey()
#
# cap = cv2.VideoCapture("./test.mp4")
# while not cap.isOpened():
#     cap = cv2.VideoCapture("./test.mp4")
#     cv2.waitKey(1000)
#     print("Wait for the header")
#
# pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
# # get video duration
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count / fps
# print(f"Duration: {duration:.2f} seconds")
#
#
# def method_name():
#     global pos_frame, sheet_mat, coordinates
#     while True:
#         flag, frame = cap.read()
#         if flag:
#             try:
#                 sheet_mat, coordinates = process_image(frame)
#                 if sheet_mat is not None:
#                     cv2.imshow('Sheet', sheet_mat)
#             except Exception as e:
#                 pass
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#         else:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
#             print("frame is not ready")
#             cv2.waitKey(1000)
#         if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
#             print("End of video")
#             break
#
# duration = timeit.Timer('method_name()', globals=globals()).timeit(1)
# print(f"Duration: {duration / 1:.2f} seconds")



