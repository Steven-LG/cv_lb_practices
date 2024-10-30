import cv2
import numpy as np

# XYZ reference values (D65 illuminant)
Xn = 95.047
Yn = 100.000
Zn = 108.883

def f(t):
    return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)

# Function to convert RGB to Lab using the provided formulas
def rgb_to_lab(image):
    # Step 1: Convert RGB image to XYZ color space
    img_xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

    # Step 2: Normalize the X, Y, and Z values
    X = img_xyz[:, :, 0] / 255 * 100  # Normalize to percentage
    Y = img_xyz[:, :, 1] / 255 * 100
    Z = img_xyz[:, :, 2] / 255 * 100

    # Step 3: Apply the formulas for L, a, and b
    L = np.where((Y / Yn) > 0.008856, 116 * np.cbrt(Y / Yn) - 16, 903.3 * (Y / Yn))
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))

    return L, a, b

def detect_powdery_mildew(image_path):
    # Step 1: Load the RGB color image
    img = cv2.imread(image_path)
    
    # Step 2: Convert to LAB color space
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = rgb_to_lab(img)
    
    # Step 3: Select 'a' channel
    # a_channel = lab[:, :, 1]
    
    # Step 4: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a_channel_cleaned = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    a_channel_eroded = cv2.erode(a_channel_cleaned, kernel, iterations=1)

    _, binary_mask_a = cv2.threshold(a_channel_eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_a_channel = cv2.bitwise_and(a.astype(np.uint8), a.astype(np.uint8), mask=binary_mask_a)

    # blue_channel = img[:, :, 0]
    b,_,_ = cv2.split(img)

    max_intensity = np.max(b)
    Th = max_intensity - 0.15 * max_intensity  # Threshold calculation
    _, binary_mask_disease = cv2.threshold(b, Th, 255, cv2.THRESH_BINARY)

    # max_a = np.max(a_channel_eroded)
    # threshold_value = max_a - 0.15 * max_a
    # _, binary_mask_a = cv2.threshold(a_channel_eroded, threshold_value, 255, cv2.THRESH_BINARY)


    
    
    # Display results
    cv2.imshow('Original Image', img)
    cv2.imshow('a', a_channel_cleaned)
    cv2.imshow('blue', b)
    cv2.imshow('Binary Mask (a Channel)', masked_a_channel)
    cv2.imshow('Binary Mask (disease)', binary_mask_disease)
    # cv2.imshow('Binary Mask (a Channel)', binary_mask_a)

    # cv2.imshow('Disease Mask')
    # cv2.imshow('Combined Mask')
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
base_folder = 'final_project/'
detect_powdery_mildew(f'{base_folder}test.png')