import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

base_practice_folder = 'fourth_practice/'
csv_output_folder = f'{base_practice_folder}/csv_output'
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)

histogram_output_folder = f'{base_practice_folder}/histograms'
if not os.path.exists(histogram_output_folder):
    os.makedirs(histogram_output_folder)

def save_channel_to_csv(channel, filename, fmt='%.8f'):
    filepath = os.path.join(csv_output_folder, filename)
    np.savetxt(filepath, channel, delimiter=',', fmt=fmt)

# Get an image and retrieve its intensities
image = cv2.imread(f'{base_practice_folder}/image.png')
segment_height = 40
segment_width = 62

start_row, start_col = 100, 150

end_row = start_row + segment_height
end_col = start_col + segment_width

# intensity_matrix = cv2.cvtColor(image[start_row:end_row, start_col:end_col], cv2.COLOR_BGR2GRAY)
intensity_matrix = np.array([
    [153, 175, 160, 176, 159, 149],
    [167, 163, 158, 155, 162, 166],
    [171, 171, 153, 153, 179, 159],
    [167, 165, 163, 176, 152, 170],
    [160, 152, 179, 151, 179, 147],
    [158, 173, 176, 162, 167, 157],
])

sorted_normalized_intensity_matrix = np.sort(intensity_matrix, axis=None).reshape(intensity_matrix.shape)
unique_sorted_normalized_intensity_matrix = np.unique(sorted_normalized_intensity_matrix)

print(unique_sorted_normalized_intensity_matrix)
print(sorted_normalized_intensity_matrix)

min_intensity = np.min(sorted_normalized_intensity_matrix)
max_intensity = np.max(sorted_normalized_intensity_matrix)
bins = np.arange(min_intensity, max_intensity + 2) - 0.5

counts, bins, patches = plt.hist(sorted_normalized_intensity_matrix.ravel(), bins=bins, color='blue', alpha=0.7)
plt.xticks(np.arange(min_intensity, max_intensity + 1), rotation=90)
plt.title('Histograma de Intensidades')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')

plt.savefig(f'{histogram_output_folder}/segment_deviations.png')

# Calcular la frecuencia de ocurrencia de cada intensidad
unique, counts = np.unique(sorted_normalized_intensity_matrix, return_counts=True)
frequency = dict(zip(unique, counts))

# Normalizar la frecuencia por el tamaño de la matriz
total_elements = sorted_normalized_intensity_matrix.size
normalized_intensity_frequency = {k: int(v) for k, v in frequency.items()}
normalized_intensity_frequency_matrix = np.vectorize(normalized_intensity_frequency.get)(sorted_normalized_intensity_matrix)

print(f'Frecuencia de normalizada de ocurrencia: {normalized_intensity_frequency_matrix}')
print(intensity_matrix.size)

# Ordenar la matriz de frecuencias de intensidad normalizada de forma ascendente
# sorted_normalized_frequency_matrix = np.sort(normalized_intensity_frequency_matrix, axis=None).reshape(sorted_normalized_intensity_matrix.shape)

# print(f'Matriz de frecuencias de intensidad normalizada ordenada:\n{sorted_normalized_frequency_matrix}')

# print(normalized_intensity_frequency_matrix[0][0])
# print(sorted_normalized_intensity_matrix.size)
# print(normalized_intensity_frequency_matrix[0][0] / sorted_normalized_intensity_matrix.size)

ocurrence_frequency = normalized_intensity_frequency_matrix / intensity_matrix.size
print(f'Frecuencia de ocurrencia: {ocurrence_frequency}')

intensity_threshold = 162 #131 es el optimo

# sorted
thresholded_segment = intensity_matrix < intensity_threshold
first_belonging_matrix = np.where(thresholded_segment, 1, 0)
second_belonging_matrix = np.where(thresholded_segment, 0, 1)

print(f'Matriz de pertenencia 1: {first_belonging_matrix}')
print(f'Matriz de pertenencia 2: {second_belonging_matrix}')

flattened_first_belonging = first_belonging_matrix.ravel()
flattened_second_belonging = second_belonging_matrix.ravel()
flattened_ocurrence_frequency = ocurrence_frequency.ravel()

ocurrence_frequency_of_first_belonging = flattened_first_belonging * flattened_ocurrence_frequency
ocurrence_frequency_of_second_belonging = np.dot(flattened_second_belonging, flattened_ocurrence_frequency)

print(f'Suma de productos de los vectores de frecuencia y pertenencia (primera matriz de pertenencia): {ocurrence_frequency_of_first_belonging[:16]}')
print(f'Suma de productos de los vectores de frecuencia y pertenencia (segunda matriz de pertenencia): {ocurrence_frequency_of_second_belonging}')

# # print(f'First belonging matrix ocurrence: {ocurrence_frequency_of_first_belonging/np.sum(ocurrence_frequency) * 100}%')
# print(np.sum(ocurrence_frequency))
# print(f'First belonging matrix ocurrence: {ocurrence_frequency_of_first_belonging}')
# print(f'Second belonging matrix ocurrence: {ocurrence_frequency_of_second_belonging}')

# print(f'Second belonging matrix ocurrence: {ocurrence_frequency_of_second_belonging/np.sum(ocurrence_frequency) * 100}%')

# # Mean of first belonging matrix
# mean_first_belonging_matrix = np.sum(intensity_matrix * ocurrence_frequency * first_belonging_matrix) / ocurrence_frequency_of_first_belonging
# print(f"Mean of first: {mean_first_belonging_matrix}")

# mean_second_belonging_matrix = np.sum(intensity_matrix * ocurrence_frequency * second_belonging_matrix) / ocurrence_frequency_of_second_belonging
# print(f"Mean of second: {mean_second_belonging_matrix}")


# image_frequency = np.sum(ocurrence_frequency * intensity_matrix) / np.sum(ocurrence_frequency_of_first_belonging + ocurrence_frequency_of_second_belonging)
# print(f"Image frequency: {image_frequency}")

# # Calcular la varianza de las intensidades para cada matriz de pertenencia
# variance_first_belonging = np.sum(((intensity_matrix - mean_first_belonging_matrix) ** 2) * first_belonging_matrix) / np.sum(first_belonging_matrix)
# variance_second_belonging = np.sum(((intensity_matrix - mean_second_belonging_matrix) ** 2) * second_belonging_matrix) / np.sum(second_belonging_matrix)

# print(f"Varianza de first: {variance_first_belonging}")
# print(f"Varianza de second: {variance_second_belonging}")

# total_elements = intensity_matrix.size
# variance_binarized = (np.sum(first_belonging_matrix) / total_elements) * variance_first_belonging + (np.sum(second_belonging_matrix) / total_elements) * variance_second_belonging

# print(f"Varianza binarizada: {variance_binarized}")

# # Varianza de la distribución de intensidad
# intensity_variance = np.sum((intensity_matrix - (np.sum(intensity_matrix) / intensity_matrix.size)) ** 2) / intensity_matrix.size
# print(f"Varianza de intensidad: {intensity_variance}")

# # relacion de varianzas
# rv = variance_binarized / intensity_variance
# print(f"Relación de varianzas: {rv}")


# # cv2.imshow('Original Image', intensity_matrix)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
