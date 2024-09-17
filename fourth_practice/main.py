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

intensity_matrix = cv2.cvtColor(image[start_row:end_row, start_col:end_col], cv2.COLOR_BGR2GRAY)

print(intensity_matrix)

counts, bins, patches = plt.hist(intensity_matrix.ravel(), bins=50, color='blue', alpha=0.7)
# plt.xticks(bins, rotation=90)
plt.title('Histograma de Intensidades')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
plt.savefig(f'{histogram_output_folder}/segment_deviations.png')

save_channel_to_csv(intensity_matrix, 'original_segment.csv', fmt='%d')

# P_i = n_i / N
ocurrence_frequency = intensity_matrix / intensity_matrix.size
intensity_threshold = 131 #131 es el optimo

thresholded_segment = intensity_matrix < intensity_threshold
first_belonging_matrix = np.where(thresholded_segment, 1, 0)
second_belonging_matrix = np.where(thresholded_segment, 0, 1)

ocurrence_frequency_of_first_belonging = np.sum(ocurrence_frequency * first_belonging_matrix)
ocurrence_frequency_of_second_belonging = np.sum(ocurrence_frequency * second_belonging_matrix)

# print(f'First belonging matrix ocurrence: {ocurrence_frequency_of_first_belonging/np.sum(ocurrence_frequency) * 100}%')
print(f'First belonging matrix ocurrence: {ocurrence_frequency_of_first_belonging}')
print(f'Second belonging matrix ocurrence: {ocurrence_frequency_of_second_belonging/np.sum(ocurrence_frequency) * 100}%')

# sum_of_first_belonging_matrix = np.sum(first_belonging_matrix)
# print(f'Sum of first belonging matrix: {sum_of_first_belonging_matrix}')

# sum_of_second_belonging_matrix = np.sum(second_belonging_matrix)
# print(f'Sum of second belonging matrix: {sum_of_second_belonging_matrix}')

# # Sumar las dos matrices de pertenencia
# sum_of_belonging_matrices = np.sum(first_belonging_matrix + second_belonging_matrix)

# # 100% de las intensidades
# sum_of_ocurrences_frequency = np.sum(ocurrence_frequency)

# if np.isclose(sum_of_belonging_matrices, sum_of_ocurrences_frequency):
#     print("La suma de las dos matrices de pertenencia cubre el 100% de las intensidades.")
# else:
#     print("La suma de las dos matrices de pertenencia NO cubre el 100% de las intensidades.")

# Mean of first belonging matrix
mean_first_belonging_matrix = np.sum(intensity_matrix * ocurrence_frequency * first_belonging_matrix) / ocurrence_frequency_of_first_belonging
print(f"Mean of first: {mean_first_belonging_matrix}")

mean_second_belonging_matrix = np.sum(intensity_matrix * ocurrence_frequency * second_belonging_matrix) / ocurrence_frequency_of_second_belonging
print(f"Mean of second: {mean_second_belonging_matrix}")

# first_belonging_matrix_calculated_mean = np.sum((image_segment * ocurrence_frequency * first_belonging_matrix)) / mean_first_belonging_matrix
# print(f"Calculated mean of first: {first_belonging_matrix_calculated_mean}")

# first_belonging_matrix_calculated_mean = np.sum((image_segment * ocurrence_frequency * second_belonging_matrix)) / mean_second_belonging_matrix
# print(f"Calculated mean of first: {first_belonging_matrix_calculated_mean}")

image_frequency = np.sum(ocurrence_frequency * intensity_matrix) / np.sum(ocurrence_frequency_of_first_belonging + ocurrence_frequency_of_second_belonging)
print(f"Image frequency: {image_frequency}")


#Varianza binarizada
# d = (ocurrence_frequency_of_first_belonging*(mean_first_belonging_matrix - image_frequency)**2) + (ocurrence_frequency_of_second_belonging * (mean_second_belonging_matrix-image_frequency)**2)
# print(f'Varianza binarizada {d}')

# Calcular la varianza de las intensidades para cada matriz de pertenencia
variance_first_belonging = np.sum(((intensity_matrix - mean_first_belonging_matrix) ** 2) * first_belonging_matrix) / np.sum(first_belonging_matrix)
variance_second_belonging = np.sum(((intensity_matrix - mean_second_belonging_matrix) ** 2) * second_belonging_matrix) / np.sum(second_belonging_matrix)

print(f"Varianza de first: {variance_first_belonging}")
print(f"Varianza de second: {variance_second_belonging}")

total_elements = intensity_matrix.size
variance_binarized = (np.sum(first_belonging_matrix) / total_elements) * variance_first_belonging + (np.sum(second_belonging_matrix) / total_elements) * variance_second_belonging

print(f"Varianza binarizada: {variance_binarized}")



# Varianza de la distribución de intensidad
# intensity_variance = np.sum(((intensity_matrix - image_frequency) ** 2) * ocurrence_frequency) / intensity_matrix.size
intensity_variance = np.sum((intensity_matrix - (np.sum(intensity_matrix) / intensity_matrix.size)) ** 2) / intensity_matrix.size
# intensity_variance = ((intensity_matrix - image_frequency) ** 2) * ocurrence_frequency

print(f"Varianza de intensidad: {intensity_variance}")


# relacion de varianzas
rv = variance_binarized / intensity_variance
print(f"Relación de varianzas: {rv}")


cv2.imshow('Original Image', intensity_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()
