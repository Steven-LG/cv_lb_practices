import cv2
import numpy as np
import os
import random

# Base Project Preparation

base_folder = 'final_project/'
dataset = 'dataset/'
condition = 'disease'

# List all files in the directory
dir = base_folder+dataset+condition
files = os.listdir(dir)

# Filter image files (e.g., jpg, png)
image_files = [f for f in files if f.endswith(('.JPG', '.png', '.jpeg', '.bmp', '.tiff'))]

# Sort the image files
# image_files.sort()

# Get the first image file
image_name = random.choice(image_files)

image = cv2.imread(f'{base_folder}{dataset}{condition}/{image_name}')

## Background removal ##
    
# RGB to LAB Conversion
def f(t):
    """Helper function for the transformation used in Lab conversion."""
    return np.where(t > 0.008856, np.cbrt(t), (t * 903.3) / 100)

def rgb_to_lab(image):
    # Step 1: Convert RGB image to XYZ color space
    img_xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

    # Step 2: Normalize the X, Y, and Z values
    X = img_xyz[:, :, 0] / 255 * 100  # Normalize to percentage
    Y = img_xyz[:, :, 1] / 255 * 100
    Z = img_xyz[:, :, 2] / 255 * 100

    Xn = 95.047
    Yn = 100.000
    Zn = 108.883

    # Step 3: Calculate L value with two conditions
    L = np.where((Y / Yn) > 0.008856, 
                   116 * np.cbrt(Y / Yn) - 16, 
                   903.3 * (Y / Yn))  # L value based on the first condition
    
    # Step 4: Calculate a and b values
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))

    return L, a, b  # Return both L values as a tuple

L, a, b = rgb_to_lab(image)
inverted_a = cv2.bitwise_not(a)

# Create a structuring element (disk shape with size 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
filled_mask = cv2.morphologyEx(inverted_a, cv2.MORPH_CLOSE, kernel)
eroded_mask = cv2.erode(filled_mask, kernel, iterations=1)
binary_mask = cv2.morphologyEx(inverted_a, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))


blue_channel = image[:, :, 0]  # Blue channel is at index 0 in BGR format
cv2.imshow("blue channel", blue_channel)
# brightness_increase = 50  # Adjust this value for slight enhancement
# enhanced_blue_channel = cv2.add(blue_channel, brightness_increase)

blue_channel = cv2.bitwise_and(blue_channel.astype(np.uint8), blue_channel.astype(np.uint8), mask=binary_mask.astype(np.uint8))

threshold_value = np.max(blue_channel)-0.15*np.max(blue_channel)
_, binary_mask_disease = cv2.threshold(blue_channel, threshold_value, 255, cv2.THRESH_BINARY)

red_image = np.zeros_like(image)
red_image[:, :, 2] = 255  # Set the red channel to 255
final_image = np.where(binary_mask_disease[:, :, np.newaxis] == 255, red_image, image)

cv2.imshow('Original Image', image)
# cv2.imshow('L', L)
cv2.imshow('a', inverted_a)
# cv2.imshow('b', b)
# cv2.imshow('filled', filled_mask)
# cv2.imshow('eroded', eroded_mask)
cv2.imshow('binary_mask', binary_mask)
# cv2.imshow("blue channel", blue_channel)
cv2.imshow('binary_mask_disease', binary_mask_disease)
cv2.imshow("final", final_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

print()



# Select 'a' Channel
# Apply morphological operations
# Create Binary Mask


## Segmentation of Disease Affected Area ##

# From Binary Mask, select Blue Channel
# Intensity Adjustment
# Adaptive Intensity Based Thresholding
# Disease Area Segmented


## Quantification of Disease Area ##

# From Disease Area Segmented, calculate Total Area and Diseases affected Area
# Calculate ratio of total disease affected area to total leaf area