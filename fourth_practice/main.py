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

images_output_folder = f'{base_practice_folder}/images'
if not os.path.exists(images_output_folder):
    os.makedirs(images_output_folder)

def save_channel_to_csv(channel, filename, fmt='%.8f'):
    filepath = os.path.join(csv_output_folder, filename)
    np.savetxt(filepath, channel, delimiter=',', fmt=fmt)

image = cv2.imread(f'{base_practice_folder}/image.png')

log_process = False

intensity_matrix = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# intensity_matrix = np.array([
#     [153, 175, 160, 176, 159, 149],
#     [167, 163, 158, 155, 162, 166],
#     [171, 171, 153, 153, 179, 159],
#     [167, 165, 163, 176, 152, 170],
#     [160, 152, 179, 151, 179, 147],
#     [158, 173, 176, 162, 167, 157],
# ])

sorted_normalized_intensity_matrix = np.sort(intensity_matrix, axis=None).reshape(intensity_matrix.shape)
unique_sorted_normalized_intensity_matrix = np.unique(sorted_normalized_intensity_matrix)

save_intensity_histogram = True

if save_intensity_histogram:
    min_intensity = np.min(sorted_normalized_intensity_matrix)
    max_intensity = np.max(sorted_normalized_intensity_matrix)

    counts, bins, patches = plt.hist(sorted_normalized_intensity_matrix.ravel(), bins=30, color='blue', alpha=0.7)
    
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

flattened_intensity_matrix = sorted_normalized_intensity_matrix.flatten()
flattened_frequency_matrix = normalized_intensity_frequency_matrix.flatten()
unique_values, indices = np.unique(flattened_intensity_matrix, return_index=True)
unique_frequencies = flattened_frequency_matrix[indices]
unique_probabilities = unique_frequencies / normalized_intensity_frequency_matrix.size

if log_process:
    print(unique_sorted_normalized_intensity_matrix)
    print(sorted_normalized_intensity_matrix)
    print(f'Frecuencia de normalizada de ocurrencia: {normalized_intensity_frequency_matrix}')
    print(intensity_matrix.size)

print(f'Resultados: {unique_values}')
print(f'Frecuencias unicas: {unique_frequencies}')
print(f'Frecuencias de aparición: {unique_probabilities}')

image_mean = np.sum(unique_values * unique_probabilities) #uT

def calculate_variance_relation(image_mean, unique_probabilities, unique_values, intensity_threshold=162):
    thresholded_segment = unique_values <= intensity_threshold

    first_belonging_matrix = np.where(thresholded_segment, 1, 0)
    second_belonging_matrix = np.where(thresholded_segment, 0, 1)

    flattened_first_belonging = first_belonging_matrix.ravel()
    flattened_second_belonging = second_belonging_matrix.ravel()

    ocurrence_frequency_of_first_belonging = np.sum(flattened_first_belonging * unique_probabilities)
    ocurrence_frequency_of_second_belonging = np.sum(flattened_second_belonging * unique_probabilities)

    if ocurrence_frequency_of_first_belonging == 0 or ocurrence_frequency_of_second_belonging == 0:
        return float('inf')  # Evitar división por cero
    
    mean_first_belonging_matrix = np.sum(unique_values * unique_probabilities * flattened_first_belonging) / ocurrence_frequency_of_first_belonging #u0
    mean_second_belonging_matrix = np.sum(unique_values * unique_probabilities * flattened_second_belonging) / ocurrence_frequency_of_second_belonging #u1

    binarized_variance = (ocurrence_frequency_of_first_belonging*(mean_first_belonging_matrix - image_mean)**2) + (ocurrence_frequency_of_second_belonging*(mean_second_belonging_matrix - image_mean)**2) 

    image_variance = (unique_values-image_mean)**2 * unique_probabilities

    variance_relationship = binarized_variance / np.sum(image_variance) #n = d²_b / d²_T

    if log_process:
        print(f'Suma de productos de los vectores de frecuencia y pertenencia (primera matriz de pertenencia): {ocurrence_frequency_of_first_belonging}')
        print(f'Suma de productos de los vectores de frecuencia y pertenencia (segunda matriz de pertenencia): {ocurrence_frequency_of_second_belonging}')
        print(f'Media de la primera matriz de pertenencia: {mean_first_belonging_matrix}')
        print(f'Media de la segunda matriz de pertenencia: {mean_second_belonging_matrix}')
        print(f'Varianza binarizada: {binarized_variance}')
        print(f'Varianza de la imagen: {np.sum(image_variance)}')
        print(f'Relación de varianzas: {variance_relationship}')
        print('\n')

    return variance_relationship
    
def find_optimal_threshold(sorted_normalized_intensity_matrix):
    min_intensity = int(np.min(sorted_normalized_intensity_matrix))
    max_intensity = int(np.max(sorted_normalized_intensity_matrix))
    optimal_threshold = min_intensity
    min_rv = float('inf')

    for threshold in range(min_intensity, max_intensity + 1):
        print(f'Resultados para umbral con intensidad: {threshold}')
        rv = calculate_variance_relation(image_mean=image_mean, unique_probabilities=unique_probabilities, unique_values=unique_values, intensity_threshold=threshold)
        if rv < min_rv:
            min_rv = rv
            optimal_threshold = threshold

    return optimal_threshold, min_rv

def calculate_entropy(probabilities):
    probabilities = probabilities[probabilities > 0]  # Eliminar ceros para evitar log(0)
    return -probabilities * np.log10(probabilities)

def calculate_total_entropy(intensity_threshold=162):
    thresholded_segment = unique_values <= intensity_threshold

    first_belonging_matrix = np.where(thresholded_segment, 1, 0).ravel()
    second_belonging_matrix = np.where(thresholded_segment, 0, 1).ravel()

    ocurrence_frequency_of_first_belonging = np.sum(first_belonging_matrix * unique_probabilities)
    ocurrence_frequency_of_second_belonging = np.sum(second_belonging_matrix * unique_probabilities)

    if ocurrence_frequency_of_first_belonging == 0 or ocurrence_frequency_of_second_belonging == 0:
        return float('inf')  # Evitar división por cero
    
    first_belonging_probabilities = calculate_entropy(ocurrence_frequency_of_first_belonging)
    second_belonging_probabilities = calculate_entropy(ocurrence_frequency_of_second_belonging)

    print(f'Entropía del primer bloque {first_belonging_probabilities}')
    print(f'Entropía del segundo bloque {second_belonging_probabilities}')

    return np.sum(first_belonging_probabilities + second_belonging_probabilities)

def find_optimal_threshold_min_entropy(intensities):
    min_intensity = int(np.min(intensities))
    max_intensity = int(np.max(intensities))
    optimal_threshold = min_intensity
    min_entropy = float('inf')

    for threshold in range(min_intensity, max_intensity + 1):
        print(f'Resultados para umbral con intensidad: {threshold}')
        total_entropy = calculate_total_entropy(intensity_threshold=threshold)
        print(f'Entropía total: {total_entropy}')
        if total_entropy < min_entropy:
            print(f'Entropía mínima: {total_entropy}')
            print(f'Umbral óptimo: {threshold}')
            min_entropy = total_entropy
            optimal_threshold = threshold
        print(f'\n')
        
    return optimal_threshold, min_entropy    

def calculate_global_valley(unique_values, unique_probabilities, image_mean):
    image_variance = (unique_values-image_mean)**2 * unique_probabilities
    standard_deviation = np.sqrt(np.sum(image_variance))

    max_value = np.array([])
    best_threshold = 0
    for _, index in enumerate(range(0, unique_frequencies.shape[0]-2)):
        print(f'Resultados para umbral con intensidad: {unique_values[index+1]}')
        weighted_frequency_difference = ((standard_deviation*(unique_frequencies[index]-unique_frequencies[index+1])) + (standard_deviation*(unique_frequencies[index+2]-unique_frequencies[index+1]))) / 2    
        print(f'Diferencia de frecuencia ponderada: {weighted_frequency_difference}')
        if max_value.size == 0 or np.abs(weighted_frequency_difference) > np.max(max_value):
            best_threshold = unique_values[index+1]
            print(f'Nuevo mejor umbral: {best_threshold}')
        
        max_value = np.append(max_value, np.abs(weighted_frequency_difference))
        print(f'\n')

    return (np.max(max_value), best_threshold)

print(f'Optimización de umbral por relación de varianzas')
optimal_threshold, min_variance_relation = find_optimal_threshold(sorted_normalized_intensity_matrix=sorted_normalized_intensity_matrix)
print(f'Optimal threshold {optimal_threshold}')
print(f'Min variance relation {min_variance_relation}')

optimal_threshold, min_entropy = find_optimal_threshold_min_entropy(intensities=unique_values)
print(f'Entropía: {calculate_total_entropy()}')
print(f'Umbral optimo: {optimal_threshold}')
print(f'Entropía minima: {min_entropy}')

print(f'Optimización de umbral por valle global')
max_value, best_threshold = calculate_global_valley(unique_values, unique_probabilities, image_mean)
print(f'Mejor valor: {max_value}')
print(f'Mejor umbral: {best_threshold}')


def reduce_tones(image, num_tones):
    thresholds = np.linspace(0, 256, num_tones + 1)
    reduced_image = np.zeros_like(image)

    print(f'Para numero de tonos {num_tones}')
    print(f'Umbrales {thresholds}')
    print('\n')
    for i in range(num_tones):
        lower_bound = thresholds[i]
        upper_bound = thresholds[i + 1]
    
        print(f'Lower bound: {lower_bound}')
        print(f'Upper bound: {upper_bound}')
        print('\n')

        mask = (image >= lower_bound) & (image < upper_bound)
        reduced_image[mask] = (lower_bound + upper_bound) // 2

    return reduced_image

three_tone_img = reduce_tones(intensity_matrix, 3).astype(np.uint8)
four_tone_img = reduce_tones(intensity_matrix, 4).astype(np.uint8)
eight_tone_img = reduce_tones(intensity_matrix, 8).astype(np.uint8)
sixteen_tone_img = reduce_tones(intensity_matrix, 16).astype(np.uint8)

factor_scale = 75/100
cv2.imshow('3 tone', cv2.resize(three_tone_img, (0,0), fx=factor_scale, fy=factor_scale))
cv2.imshow('4 tone', cv2.resize(four_tone_img, (0,0), fx=factor_scale, fy=factor_scale))
cv2.imshow('8 tone', cv2.resize(eight_tone_img, (0,0), fx=factor_scale, fy=factor_scale))
cv2.imshow('16 tone', cv2.resize(sixteen_tone_img, (0,0), fx=factor_scale, fy=factor_scale))

cv2.imwrite(f'{images_output_folder}/three_tone_img.png', three_tone_img)
cv2.imwrite(f'{images_output_folder}/four_tone_img.png', four_tone_img)
cv2.imwrite(f'{images_output_folder}/eight_tone_img.png', eight_tone_img)
cv2.imwrite(f'{images_output_folder}/sixteen_tone_img.png', sixteen_tone_img)

save_channel_to_csv(three_tone_img, 'three_tone_img.csv', fmt='%d')
save_channel_to_csv(four_tone_img, 'four_tone_img.csv', fmt='%d')
save_channel_to_csv(eight_tone_img, 'eight_tone_img.csv', fmt='%d')
save_channel_to_csv(sixteen_tone_img, 'sixteen_tone_img.csv', fmt='%d')

cv2.waitKey(0)
cv2.destroyAllWindows()