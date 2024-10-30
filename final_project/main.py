import cv2

base_folder = 'final_project/'
image = cv2.imread(f'{base_folder}test.png')

if image is None:
    print('Error: No se pudo cargar la imagen')

image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
a_channel = image_lab[:, :, 1]

print(a_channel)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
a_channel_cleaned = cv2.morphologyEx(a_channel, cv2.MORPH_CLOSE, kernel)

_, binary_mask = cv2.threshold(a_channel_cleaned, 128, 255, cv2.THRESH_BINARY)

b_channel = image_lab[:, :, 2]

max_intensity_b = b_channel.max()
th = max_intensity_b - 0.15 * max_intensity_b
# _, disease_mask = cv2.threshold(b_channel, th, 255, cv2.THRESH_BINARY)
b_channel_eq = cv2.equalizeHist(b_channel)
disease_mask = cv2.adaptiveThreshold(b_channel_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, leaf_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


disease_area = cv2.countNonZero(disease_mask)  # Count disease-affected pixels
healthy_area = cv2.countNonZero(binary_mask) - disease_area  # Remaining pixels are healthy
ratio = disease_area / healthy_area if healthy_area > 0 else 0  # Ratio of disease to healthy area

# Print the result
print(f"Ratio of diseased area to healthy area: {ratio}")

cv2.imshow("Original Image", image)
cv2.imshow("Leaf Mask", leaf_mask)
# cv2.imshow("Binary Mask", binary_mask)
cv2.imshow("Disease Mask", disease_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()