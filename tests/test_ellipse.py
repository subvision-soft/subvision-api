import cv2
import os
import numpy as np
import math

# Global variables
points = []
ellipse_points = []
current_image = None
cropped_image = None
image_index = 0
image_files = []
zoom_scale = 1.0  # Zoom scale factor

# Helper functions
def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def to_degrees(radians):
    return math.degrees(radians)

# Mouse callback function for selecting cropping region
def select_region(event, x, y, flags, param):
    global points, current_image, zoom_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust coordinates for zoom
        x = int(x / zoom_scale)
        y = int(y / zoom_scale)
        points.append((x, y))
        if len(points) <= 4:
            # Draw the point on the image
            cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                # Draw a line between the last two points
                cv2.line(current_image, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                # Draw a line between the first and last points to close the shape
                cv2.line(current_image, points[-1], points[0], (0, 255, 0), 2)
            # Display the image with zoom
            display_image = cv2.resize(current_image, None, fx=zoom_scale, fy=zoom_scale)
            cv2.imshow("Image", display_image)

# Mouse callback function for drawing ellipses
def draw_ellipse(event, x, y, flags, param):
    global ellipse_points, cropped_image, zoom_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust coordinates for zoom
        x = int(x / zoom_scale)
        y = int(y / zoom_scale)
        ellipse_points.append((x, y))
        if len(ellipse_points) == 3:
            # Calculate ellipse parameters
            center = ellipse_points[0]
            height = get_distance(center, ellipse_points[1]) * 2
            width = get_distance(center, ellipse_points[2]) * 2
            angle = to_degrees(get_angle(center, ellipse_points[1])) - 90
            # Draw the ellipse on the cropped image
            cv2.ellipse(cropped_image, (center, (int(width), int(height)), angle), (0, 255, 0), 2)
            # Display the image with zoom
            display_image = cv2.resize(cropped_image, None, fx=zoom_scale, fy=zoom_scale)
            cv2.imshow("Cropped Image", display_image)
            ellipse_points.clear()

def process_images(folder_path):
    global current_image, cropped_image, image_index, image_files, points, ellipse_points, zoom_scale

    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    if total_images == 0:
        print("No images found in the folder.")
        return

    # Iterate through all images
    while image_index < total_images:
        image_path = os.path.join(folder_path, image_files[image_index])
        current_image = cv2.imread(image_path)
        original_image = current_image.copy()  # Keep a clean copy for cropping
        points.clear()

        # Display the image with zoom
        display_image = cv2.resize(current_image, None, fx=zoom_scale, fy=zoom_scale)
        cv2.imshow("Image", display_image)
        cv2.setMouseCallback("Image", select_region)

        print(f"Processing image {image_index + 1}/{total_images}: {image_files[image_index]}")
        print("Click four points to define the cropping region. Press 'c' to crop or 'n' to skip.")
        print("Press '+' to zoom in, '-' to zoom out.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(points) == 4:
                    break
                else:
                    print("Please click exactly four points to define the cropping region.")
            elif key == ord('n'):  # Skip to the next image
                points.clear()
                break
            elif key == ord('+'):  # Zoom in
                zoom_scale = min(zoom_scale + 0.1, 3.0)  # Limit zoom to 3x
                display_image = cv2.resize(current_image, None, fx=zoom_scale, fy=zoom_scale)
                cv2.imshow("Image", display_image)
            elif key == ord('-'):  # Zoom out
                zoom_scale = max(zoom_scale - 0.1, 0.1)  # Limit zoom to 0.5x
                display_image = cv2.resize(current_image, None, fx=zoom_scale, fy=zoom_scale)
                cv2.imshow("Image", display_image)
            elif key == 27:  # ESC key to exit
                cv2.destroyAllWindows()
                return

        if len(points) == 4:
            # Perform perspective transform to crop the region
            pts = np.array(points, dtype=np.float32)
            # Define the width and height of the cropped image
            width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
            height = max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0]))
            # Define the destination points for the perspective transform
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            # Compute the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(pts, dst)
            # Apply the perspective transform to the original image (without annotations)
            cropped_image = cv2.warpPerspective(original_image, matrix, (int(width), int(height)))
            # Save the cropped image
            cropped_filename = os.path.splitext(image_files[image_index])[0] + "_cropped.png"
            cropped_path = os.path.join(folder_path, cropped_filename)
            cv2.imwrite(cropped_path, cropped_image)
            print(f"Cropped image saved as {cropped_filename}")

            # Draw 5 ellipses on the cropped image
            cv2.imshow("Cropped Image", cropped_image)
            cv2.setMouseCallback("Cropped Image", draw_ellipse)
            print("Click three points to define an ellipse (center, height, width). Draw 5 ellipses.")

            ellipse_count = 0
            mask = np.zeros_like(cropped_image[:, :, 0])  # Single-channel mask for ellipses
            while ellipse_count < 5:
                key = cv2.waitKey(1) & 0xFF
                if len(ellipse_points) == 3:
                    # Calculate ellipse parameters
                    center = ellipse_points[0]
                    height = get_distance(center, ellipse_points[1]) * 2
                    width = get_distance(center, ellipse_points[2]) * 2
                    angle = to_degrees(get_angle(center, ellipse_points[1])) - 90
                    # Draw the ellipse on the mask
                    cv2.ellipse(mask, (center, (int(width), int(height)), angle), 255, -1)
                    ellipse_count += 1
                    print(f"Ellipse {ellipse_count} drawn.")
                    ellipse_points.clear()

            # Save the mask
            mask_filename = os.path.splitext(image_files[image_index])[0] + "_mask.png"
            mask_path = os.path.join(folder_path, mask_filename)
            cv2.imwrite(mask_path, mask)
            print(f"Mask saved as {mask_filename}")

        # Move to the next image
        image_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "images"  # Change this to your folder path
    process_images(folder_path)