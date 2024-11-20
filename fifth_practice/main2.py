import cv2
import numpy as np

# Load the image
base_practice_folder = 'fifth_practice/'
image = cv2.imread(f'{base_practice_folder}/image_2.jpeg')

# Convert BGR to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# blurred_mask = cv2.GaussianBlur(hsv, (11, 11), 0)
edges = cv2.Canny(image, threshold1=100, threshold2=200)

cv2.imshow('Original Image', image)
# cv2.imshow('Blurred Mask', blurred_mask)
cv2.imshow('Fungus Borders', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()