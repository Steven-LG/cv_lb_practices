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


flattened_intensity_matrix = sorted_normalized_intensity_matrix.flatten()
flattened_frequency_matrix = normalized_intensity_frequency_matrix.flatten()
unique_values, indices = np.unique(flattened_intensity_matrix, return_index=True)
unique_frequencies = flattened_frequency_matrix[indices]
unique_probabilities = unique_frequencies / normalized_intensity_frequency_matrix.size

print(f'Resultados: {unique_values}')
print(f'Frecuencias: {unique_frequencies}')
print(f'Probabilidades: {unique_probabilities}')

# TODO: SI FUNCIONA
# intensities_results = np.array([], dtype=int)
# freq_results = np.array([], dtype=int)
# p_i_results = np.array([], dtype=float)
# seen = set()

# # Recorrer la matriz ordenada y normalizada
# for i in range(sorted_normalized_intensity_matrix.shape[0]):
#     for j in range(sorted_normalized_intensity_matrix.shape[1]):
#         value = sorted_normalized_intensity_matrix[i, j]
#         if value not in seen:
#             seen.add(value)
#             freq_value = normalized_intensity_frequency_matrix[i, j]
#             freq_aparicion = freq_value / normalized_intensity_frequency_matrix.size
#             intensities_results = np.append(intensities_results, value)
#             freq_results = np.append(freq_results, freq_value)
#             p_i_results = np.append(p_i_results, freq_aparicion)

# print(f'Resultados: {intensities_results}')
# print(f'Frecuencias: {freq_results}')
# print(f'Probabilidades: {p_i_results}')
#TODO: SI FUNCIONA

image_mean = np.sum(unique_values * unique_probabilities) #uT
print(f'Media de la imagen: {image_mean}')

def calculate_variance_relation(image_mean, unique_probabilities, unique_values, intensity_threshold=162):
    thresholded_segment = unique_values <= intensity_threshold
    first_belonging_matrix = np.where(thresholded_segment, 1, 0)
    second_belonging_matrix = np.where(thresholded_segment, 0, 1)

    print(f'Matriz de pertenencia 1: {first_belonging_matrix}')
    print(f'Matriz de pertenencia 2: {second_belonging_matrix}')

    flattened_first_belonging = first_belonging_matrix.ravel()
    flattened_second_belonging = second_belonging_matrix.ravel()

    ocurrence_frequency_of_first_belonging = np.sum(flattened_first_belonging * unique_probabilities)
    ocurrence_frequency_of_second_belonging = np.sum(flattened_second_belonging * unique_probabilities)

    print(f'Suma de productos de los vectores de frecuencia y pertenencia (primera matriz de pertenencia): {ocurrence_frequency_of_first_belonging}')
    print(f'Suma de productos de los vectores de frecuencia y pertenencia (segunda matriz de pertenencia): {ocurrence_frequency_of_second_belonging}')

    mean_first_belonging_matrix = np.sum(unique_values * unique_probabilities * flattened_first_belonging) / ocurrence_frequency_of_first_belonging #u0
    mean_second_belonging_matrix = np.sum(unique_values * unique_probabilities * flattened_second_belonging) / ocurrence_frequency_of_second_belonging #u1
    print(f'Media de la primera matriz de pertenencia: {mean_first_belonging_matrix}')
    print(f'Media de la segunda matriz de pertenencia: {mean_second_belonging_matrix}')

    binarized_variance = (ocurrence_frequency_of_first_belonging*(mean_first_belonging_matrix - image_mean)**2) + (ocurrence_frequency_of_second_belonging*(mean_second_belonging_matrix - image_mean)**2) 
    print(f'Varianza binarizada: {binarized_variance}')

    image_variance = (unique_values-image_mean)**2 * unique_probabilities
    print(f'Varianza de la imagen: {np.sum(image_variance)}')

    variance_relationship = binarized_variance / np.sum(image_variance) #n = d²_b / d²_T
    print(f'Relación de varianzas: {variance_relationship}')

    return variance_relationship
    
def find_optimal_threshold(sorted_normalized_intensity_matrix):
    min_intensity = np.min(sorted_normalized_intensity_matrix)
    max_intensity = np.max(sorted_normalized_intensity_matrix)
    optimal_threshold = min_intensity
    min_rv = float('inf')

    for threshold in range(min_intensity, max_intensity + 1):
        rv = calculate_variance_relation(image_mean=image_mean, unique_probabilities=unique_probabilities, unique_values=unique_values, intensity_threshold=threshold)
        if rv < min_rv:
            min_rv = rv
            optimal_threshold = threshold

    return optimal_threshold, min_rv

optimal_threshold, min_variance_relation = find_optimal_threshold(sorted_normalized_intensity_matrix=sorted_normalized_intensity_matrix)
print(f'Optimal threshold {optimal_threshold}')
print(f'Min variance relation {min_variance_relation}')
