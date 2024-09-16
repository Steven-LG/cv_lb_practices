import os 
import numpy as np
import cv2

base_practice_folder = 'second_practice/'

img = cv2.imread(f'{base_practice_folder}/incendio.png',)
gray_scale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Limites para Shifting Horizontal/Vertical/Diagonal, Elongación/Reducción de contraste y para Aumento/Disminución de brillo
# R(x, y) = Q(x, y) - Min(Q(x, y))
# S(x, y) = (R(x, y)) / (MaxR(x, y)) 255
# MinQ(x, y), Max(Q(x, y))


# Aclarado/Oscurecimiento (P(x, y) = 1)
# Copiado (Q(x, y) = P(x, y))
# Negativo (Q(x, y) = 255 - P(x, y))
# Aumento/Disminución de brillo (Q(x, y) = P(x, y) ± β)
# Elongación de contraste (Q(x, y) = P(x, y)ɣ + β)
# Reducción de contraste (Q(x, y) = (P(x, y)/ɣ) - β)
# Shifting Horizontal/Vertical/Diagonal



output_folder = f'{base_practice_folder}/csv_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
def save_channel_to_csv(channel, filename):
    filepath = os.path.join(output_folder, filename)
    np.savetxt(filepath, channel, delimiter=',', fmt='%d')

converted_output_folder = f'{base_practice_folder}/converted_image_output'
if not os.path.exists(converted_output_folder):
    os.makedirs(converted_output_folder)
    
gray_image_path = os.path.join(converted_output_folder, 'imagen_gris.png')
cv2.imwrite(gray_image_path, gray_scale_image)

# Función para normalizar una imagen
def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    # Aplicar la fórmula: R(x, y) = Q(x, y) - Min(Q(x, y))
    R = image - min_val
    # Aplicar la fórmula: S(x, y) = (R(x, y)) / MaxR(x, y) * 255
    S = (R / (max_val - min_val)) * 255
    return np.uint8(S)

# Aumentar/Disminuir brillo (Q(x, y) = P(x, y) ± β)
def adjust_brightness(image, beta):
    brightened_image = cv2.convertScaleAbs(image, alpha=1, beta=beta)

    # Normalizar la imagen con brillo ajustado
    return normalize_image(brightened_image)

# Función para hacer una copia de la imagen (Q(x, y) = P(x, y))
def copy_image(image):
    return np.copy(image)

# Negativo (Q(x, y) = 255 - P(x, y))
def negative_image(image):
    return 255 - image

# Aclarado/Oscurecimiento (P(x, y) = 1)
# Aclarado
def brighten_image(image, factor=1.2):
    # Aclarar multiplicando por un factor > 1
    brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    # Normalizar la imagen aclarada
    return normalize_image(brightened_image)

# Oscurecimiento
def darken_image(image, factor=0.8):
    # Oscurecer multiplicando por un factor < 1
    darkened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    # Normalizar la imagen oscurecida
    return normalize_image(darkened_image)

# Elongación de contraste (Q(x, y) = P(x, y)ɣ + β)
def stretch_contrast(image, gamma, beta=0):
    # Ajustar el contraste utilizando gamma (para elongación o reducción)
    adjusted_image = cv2.convertScaleAbs(image, alpha=gamma, beta=beta)
    # Normalizar la imagen ajustada
    return normalize_image(adjusted_image)

# Reducción de contraste (Q(x, y) = (P(x, y)/ɣ) - β)
def reduce_contrast(image, gamma, beta=0):
    return cv2.convertScaleAbs(image, alpha=1.0/gamma, beta=-beta)

# Shifting Horizontal/Vertical/Diagonal
def shift_image(image, shift_x, shift_y):
    rows, cols = image.shape
    # Matriz de transformación para desplazamiento (shifting)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (cols, rows))
    # Normalizar la imagen desplazada
    return normalize_image(shifted_image)

# Imagen original
cv2.imshow("Original", gray_scale_image)
save_channel_to_csv(gray_scale_image, 'escala_grises.csv')

# Aclarado de la imagen (e.g., factor=1.2 para aclarar)
brightened_image = brighten_image(gray_scale_image, factor=1.2)
cv2.imshow("Brightened Image", brightened_image)
save_channel_to_csv(brightened_image, 'brightened_image.csv')

# Oscurecimiento de la imagen (e.g., factor=0.8 para oscurecer)
darkened_image = darken_image(gray_scale_image, factor=0.8)
cv2.imshow("Darkened Image", darkened_image)
save_channel_to_csv(darkened_image, 'darkened_image.csv')

copied_image = copy_image(image=gray_scale_image)
cv2.imshow("Copied Image", copied_image)
save_channel_to_csv(copied_image, 'copied_image.csv')

negative_img = negative_image(gray_scale_image)
cv2.imshow("Negative Image", negative_img)
save_channel_to_csv(negative_img, 'negative_img.csv')

# Aumento de brillo (e.g., beta=50 para aumentar el brillo)
brighter_image = adjust_brightness(gray_scale_image, beta=50)
cv2.imshow("Brighter Image", brighter_image)
save_channel_to_csv(brighter_image, 'brighter_image.csv')

# Disminución de brillo (e.g., beta=-50 para disminuir el brillo)
darker_image = adjust_brightness(gray_scale_image, beta=-50)
cv2.imshow("Darker Image", darker_image)
save_channel_to_csv(darker_image, 'darker_image.csv')

# Elongación de contraste (e.g., gamma=1.5 para elongar)
stretched_contrast_image = stretch_contrast(gray_scale_image, gamma=1.5)
cv2.imshow("Stretched Contrast Image", stretched_contrast_image)
save_channel_to_csv(stretched_contrast_image, 'stretched_contrast_image.csv')

# Reducción de contraste (e.g., gamma=0.5 para reducir)
reduced_contrast_image = stretch_contrast(gray_scale_image, gamma=0.5)
cv2.imshow("Reduced Contrast Image", reduced_contrast_image)
save_channel_to_csv(reduced_contrast_image, 'reduced_contrast_image.csv')

# Shifting horizontal y vertical (e.g., 30 píxeles a la derecha, 15 píxeles hacia abajo)
shifted_image = shift_image(gray_scale_image, shift_x=30, shift_y=15)
cv2.imshow("Shifted Image", shifted_image)
save_channel_to_csv(shifted_image, 'shifted_image.csv')

cv2.waitKey(0)
cv2.destroyAllWindows()