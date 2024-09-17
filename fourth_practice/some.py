import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def calculate_rv(intensity_matrix, intensity_threshold):
    thresholded_segment = intensity_matrix < intensity_threshold
    first_belonging_matrix = np.where(thresholded_segment, 1, 0)
    second_belonging_matrix = np.where(thresholded_segment, 0, 1)

    # Calcular la media de las intensidades para cada matriz de pertenencia
    sum_first_belonging = np.sum(first_belonging_matrix)
    sum_second_belonging = np.sum(second_belonging_matrix)

    if sum_first_belonging == 0 or sum_second_belonging == 0:
        return float('inf')  # Evitar división por cero

    mean_first_belonging = np.sum(intensity_matrix * first_belonging_matrix) / sum_first_belonging
    mean_second_belonging = np.sum(intensity_matrix * second_belonging_matrix) / sum_second_belonging

    # Calcular la varianza de las intensidades para cada matriz de pertenencia
    variance_first_belonging = np.sum(((intensity_matrix - mean_first_belonging) ** 2) * first_belonging_matrix) / sum_first_belonging
    variance_second_belonging = np.sum(((intensity_matrix - mean_second_belonging) ** 2) * second_belonging_matrix) / sum_second_belonging

    # Varianza de la distribución de intensidad
    total_elements = intensity_matrix.size
    variance_binarized = (sum_first_belonging / total_elements) * variance_first_belonging + (sum_second_belonging / total_elements) * variance_second_belonging

    # Calcular la media de las intensidades de la imagen
    image_intensity_mean = np.sum(intensity_matrix) / intensity_matrix.size

    # Calcular la varianza de las intensidades de la imagen
    intensity_variance = np.sum((intensity_matrix - image_intensity_mean) ** 2) / intensity_matrix.size

    # Relación de varianzas
    rv = variance_binarized / intensity_variance
    return rv


base_practice_folder = 'fourth_practice/'
csv_output_folder = f'{base_practice_folder}/csv_output'
histogram_output_folder = f'{base_practice_folder}/histograms'

image = cv2.imread(f'{base_practice_folder}/image.png')
segment_height = 40
segment_width = 62

start_row, start_col = 100, 150

end_row = start_row + segment_height
end_col = start_col + segment_width

intensity_matrix = cv2.cvtColor(image[start_row:end_row, start_col:end_col], cv2.COLOR_BGR2GRAY)

# # Suponiendo que intensity_matrix es tu matriz de intensidades
# intensity_matrix = np.array([[41, 41, 42, 134, 128, 117],
#                              [41, 41, 41, 189, 160, 135],
#                              [41, 42, 41, 242, 206, 181],
#                              [35, 35, 35, 45, 45, 45],
#                              [35, 35, 35, 45, 45, 45],
#                              [35, 35, 35, 43, 44, 45]])



# Guardar el histograma de desviaciones
plt.ylabel('Frecuencia')
plt.savefig(f'{histogram_output_folder}/segment_deviations.png')

def save_channel_to_csv(channel, filename, fmt='%.8f'):
    filepath = os.path.join(csv_output_folder, filename)
    np.savetxt(filepath, channel, delimiter=',', fmt=fmt)

save_channel_to_csv(intensity_matrix, 'original_segment.csv', fmt='%d')

# Búsqueda de fuerza bruta para encontrar el umbral óptimo
best_threshold = None
min_rv = float('inf')

for threshold in range(0, 256):
    rv = calculate_rv(intensity_matrix, threshold)
    if rv < min_rv:
        min_rv = rv
        best_threshold = threshold

print(f"Umbral óptimo: {best_threshold}")
print(f"Relación de varianzas mínima: {min_rv}")

# Visualizar los segmentos con el umbral óptimo
thresholded_segment = intensity_matrix < best_threshold
first_belonging_matrix = np.where(thresholded_segment, 1, 0)
second_belonging_matrix = np.where(thresholded_segment, 0, 1)

plt.subplot(1, 2, 1)
plt.title("First Belonging Matrix")
plt.imshow(first_belonging_matrix, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Second Belonging Matrix")
plt.imshow(second_belonging_matrix, cmap='gray')

plt.savefig(f'{histogram_output_folder}/segmentation_result.png')  # Guardar la figura en lugar de mostrarla

# Convertir intensity_matrix a uint8 para OpenCV
intensity_matrix_uint8 = intensity_matrix.astype(np.uint8)

# Mostrar la imagen original
cv2.imshow('Original Image', intensity_matrix_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()