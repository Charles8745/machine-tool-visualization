import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import os

# Chessboard settings
chessboard_size = (22, 16)  # Number of inner corners (cols-1, rows-1)
square_size = 0.014  # Real-world square size per grid unit
origin_x, origin_y = 0.28313, -0.73138  # Set the top-left reference point to (0.2, 0.1)
captured_image_filename = "captured_image.png"
detected_image_filename = "chessboard_detected.png"
csv_filename = "chessboard_corners.csv"

# 1️⃣ Capture an image using Intel RealSense
def capture_image():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Display the live image
            cv2.imshow('RealSense Camera', color_image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('p'):  # Press 'p' to capture
                cv2.imwrite(captured_image_filename, color_image)
                print(f"📷 Image captured and saved as {captured_image_filename}")
                break

            elif key & 0xFF == ord('q') or key == 27:  # Press 'q' or Esc to exit
                pipeline.stop()
                cv2.destroyAllWindows()
                return None
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return captured_image_filename

# 2️⃣ Detect chessboard corners (No refinement)
def detect_chessboard_corners(image_path, chessboard_size):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Draw corners on the image
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv2.imwrite(detected_image_filename, image)
        print(f"✅ Chessboard detected and saved as {detected_image_filename}")

        return corners
    else:
        print("⚠ Chessboard not detected! Ensure it is fully visible and well-lit.")
        return None

# 3️⃣ Compute real-world coordinates for each inner corner
def compute_world_coordinates(chessboard_size, square_size, origin_x, origin_y):
    world_coords = []
    
    for row in range(chessboard_size[1]):  # 16 rows (Y-axis)
        for col in range(chessboard_size[0]):  # 22 cols (X-axis)
            world_x = origin_x - col * square_size  # X increases to the right
            world_y = origin_y + row * square_size  # Y increases downward
            world_coords.append((world_x, world_y))

    return world_coords

# 4️⃣ Save pixel and real-world coordinates to a CSV file
def save_to_csv(pixel_corners, world_coords, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pixel_X", "Pixel_Y", "World_X", "World_Y"])
        for i in range(len(pixel_corners)):
            pixel = pixel_corners[i]
            world = world_coords[i]
            writer.writerow([pixel[0][0], pixel[0][1], world[0], world[1]])

    print(f"✅ Pixel and world coordinates saved to {csv_filename}")

# 🎯 **Main execution**
if __name__ == "__main__":
    image_path = capture_image()
    
    if image_path:
        pixel_corners = detect_chessboard_corners(image_path, chessboard_size)
        
        if pixel_corners is not None:
            world_coords = compute_world_coordinates(chessboard_size, square_size, origin_x, origin_y)
            save_to_csv(pixel_corners, world_coords, csv_filename)
