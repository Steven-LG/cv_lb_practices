import cv2
import numpy as np

# Load the image
base_practice_folder = 'fifth_practice/'
image = cv2.imread(f'{base_practice_folder}/image.png')

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red color range
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Create masks for red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Apply Gaussian Blur to reduce noise
blurred_mask = cv2.GaussianBlur(mask, (11, 11), 0)

# Find borders using Canny edge detection
edges = cv2.Canny(blurred_mask, threshold1=100, threshold2=200)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Red Mask', mask)
cv2.imshow('Blurred Mask', blurred_mask)
cv2.imshow('Fungus Borders', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()