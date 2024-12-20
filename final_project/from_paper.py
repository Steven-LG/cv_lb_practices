import cv2
import numpy as np
import os
import random

base_folder = 'final_project/'
dataset = 'new_dataset/'
condition = 'disease'

# List all files in the directory
dir = "/home/xsvd/Main/Homework/LBVC/final_project/new_dataset/disease/rescaled_ones"
# dir = "/home/xsvd/Main/Homework/LBVC/final_project/new_dataset/healthy/rescaled_ones"
files = os.listdir(dir)

# Filter image files (e.g., jpg, png)
image_files = [f for f in files if f.endswith(('.JPG', '.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]

# Get the first image file
image_name = random.choice(image_files)
image = cv2.imread(f'{dir}/{image_name}')

def f(t):
    """Helper function for the transformation used in Lab conversion."""
    return np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + (16 / 116))
    # return np.where(t > 0.008856, np.cbrt(t), (t * 903.3) / 100)

def rgb_to_lab(image):
    # img_xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

    # X = img_xyz[:, :, 0] / 255 * 100 
    # Y = img_xyz[:, :, 1] / 255 * 100
    # Z = img_xyz[:, :, 2] / 255 * 100

    # Xn = 95.047
    # Yn = 100.000
    # Zn = 108.883


    # Normalize BGR values to [0, 1]
    bgr = image.astype('float32') / 255.0

    # Convert BGR to RGB by reordering channels
    rgb = bgr[:, :, [2, 1, 0]]

    # RGB to XYZ conversion matrix (sRGB D65)
    matrix = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])

    # Reshape for matrix multiplication
    rgb_flat = rgb.reshape(-1, 3)

    # Convert to XYZ
    xyz_flat = np.dot(rgb_flat, matrix.T)

    # Reshape back to image shape
    xyz_image = xyz_flat.reshape(image.shape[0], image.shape[1], 3)

    X = xyz_image[:, :, 0]
    Y = xyz_image[:, :, 1]
    Z = xyz_image[:, :, 2]

    # Reference white values
    Xn = 95.047
    Yn = 100.000
    Zn = 108.883

    L = np.where((Y / Yn) > 0.008856, 
                   116 * np.cbrt(Y / Yn) - 16, 
                   903.3 * (Y / Yn))
    
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))

    return L, a, b


L, a, b = rgb_to_lab(image)
inverted_a = cv2.bitwise_not(cv2.blur(a, (5,5)))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
filled_mask = cv2.morphologyEx(inverted_a, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
eroded_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
binary_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

_, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
blue_channel_before = image[:, :, 0]
blue_channel = cv2.bitwise_and(blue_channel_before.astype(np.uint8), blue_channel_before.astype(np.uint8), mask=binary_mask.astype(np.uint8))

print()
threshold_value = np.max(blue_channel)-0.40*np.max(blue_channel)
_, binary_mask_disease = cv2.threshold(blue_channel, threshold_value, 255, cv2.THRESH_BINARY)

red_image = np.zeros_like(image)
red_image[:, :, 2] = 255
new_red = cv2.bitwise_and(red_image.astype(np.uint8), red_image.astype(np.uint8), mask=binary_mask.astype(np.uint8))

new_image = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=binary_mask.astype(np.uint8))
final_image = np.where(binary_mask_disease[:, :, np.newaxis] == 255, new_red, new_image)


cv2.imshow('Original Image', image)
# cv2.imshow('L', L)
cv2.imshow('a', inverted_a)
# cv2.imshow('b', b)
# cv2.imshow('fill', filled_mask)
# cv2.imshow('erosion', eroded_mask)
cv2.imshow('binary_mask', binary_mask)
cv2.imshow("blue channel", blue_channel)
cv2.imshow('binary_mask_disease', binary_mask_disease)
# cv2.imshow("new red", new_red)
cv2.imshow("final", final_image)

cv2.waitKey(0)

output_path = os.path.join("/home/xsvd/Main/Homework/LBVC/final_project/test", 'binary_mask_disease.JPG')
cv2.imwrite(output_path, binary_mask_disease)

white_pixel_blue_channel_count = np.sum(binary_mask < 1)
print(f'Number of white pixels in binary_mask: {white_pixel_blue_channel_count}')

white_pixel_count = np.sum(binary_mask_disease == 255)
print(f'Number of white pixels in binary_mask_disease: {white_pixel_count}')

print(f"Ratio: {white_pixel_count/white_pixel_blue_channel_count}")

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