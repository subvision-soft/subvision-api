import cv2
import os
import numpy as np

# Global variables
points = []
current_image = None
image_index = 0
image_files = []
zoom_scale = 1.0  # Zoom scale factor

# Mouse callback function
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

def process_images(folder_path):
    global current_image, image_index, image_files, points, zoom_scale

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

            # Create a black and white mask of the selected region
            mask = np.zeros_like(original_image[:, :, 0])  # Single-channel mask
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)  # Fill the selected region with white
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