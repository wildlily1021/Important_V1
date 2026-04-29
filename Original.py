import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'D:/postgraduate/AI_SIGNAL_ANA/pythonProject/photo_EXLow.jpg'
output_path = 'D:/postgraduate/AI_SIGNAL_ANA/pythonProject/photo_EXLow_nowhite.jpg'
binary_output_path = 'D:/postgraduate/AI_SIGNAL_ANA/pythonProject/photo_EXLow_binary.jpg'
black_output_path = 'D:/postgraduate/AI_SIGNAL_ANA/pythonProject/photo_EXLow_black.jpg'

image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 自定义阈值
threshold_value = 240

# 找到所有非白色点（灰度值小于阈值的点）
non_white_points = np.column_stack(np.where(gray_image < threshold_value))

# Find the bounding box of the non-white points
top_left = non_white_points.min(axis=0)
bottom_right = non_white_points.max(axis=0)

# Crop the image to the bounding box
cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

# Save the cropped image
cv2.imwrite(output_path, cropped_image)

# Get the dimensions of the cropped image
height, width, _ = cropped_image.shape

# Extract the middle row of the cropped image
middle_row = cropped_image[height // 2, :, :]

# Calculate the median of the yellow component (using the green channel as an approximation)
yellow_median = np.median(middle_row[:, 1])

# Create a binary image based on the median value
binary_image = np.where(cropped_image[:, :, 1] > yellow_median, 255, 0).astype(np.uint8)

# Save the binary image
cv2.imwrite(binary_output_path, binary_image)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)  # Use a larger kernel for more aggressive noise removal

# Opening operation (erosion followed by dilation) to remove small white noise
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Closing operation (dilation followed by erosion) to fill small black holes
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Additional opening and closing to further clean up the image
kernel_large = np.ones((7, 7), np.uint8)
cleaned_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel_large)
cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel_large)

# Analyze each row to convert based on the 50% threshold
for i in range(cleaned_image.shape[0]):
    if np.mean(cleaned_image[i, :]) > 127:  # 127 is the midpoint between 0 and 255
        cleaned_image[i, :] = 255  # Convert the entire row to white
    else:
        cleaned_image[i, :] = 0  # Convert the entire row to black

# Find contours to calculate height and center point of the white region
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    bandwidth = h
    center_frequency = center_y

    print(f"Bandwidth: {bandwidth}")
    print(f"Center Frequency: {center_frequency}")

# Find the top and bottom edges of the black part
black_top = None
black_bottom = None
for i in range(cleaned_image.shape[0]):
    if black_top is None and np.all(cleaned_image[i, :] == 0):
        black_top = i
    if np.all(cleaned_image[i, :] == 0):
        black_bottom = i

if black_top is not None and black_bottom is not None:
    print(f"Top edge of black part: {black_top}")
    print(f"Bottom edge of black part: {black_bottom}")

cv2.imwrite(black_output_path, cleaned_image)

# Show the cropped image in a separate window
plt.figure(1)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.xlim([0, width])
plt.ylim([height, 0])
plt.gca().invert_yaxis()  # Invert y-axis to have the origin at the top-left corner

# Show the cleaned binary image in a separate window
plt.figure(2)
plt.imshow(cleaned_image, cmap='gray')
plt.title('Cleaned Binary Image')
plt.xlim([0, width])
plt.ylim([height, 0])
plt.gca().invert_yaxis()  # Invert y-axis to have the origin at the top-left corner
plt.show()
