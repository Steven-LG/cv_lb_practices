import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

base_practice_folder = 'third_practice/'

csv_output_folder = f'{base_practice_folder}/csv_output'
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)

histogram_output_folder = f'{base_practice_folder}/histograms'
if not os.path.exists(histogram_output_folder):
    os.makedirs(histogram_output_folder)

def save_channel_to_csv(channel, filename, fmt='%.8f'):
    filepath = os.path.join(csv_output_folder, filename)
    np.savetxt(filepath, channel, delimiter=',', fmt=fmt)

image = cv2.imread(f'{base_practice_folder}/image.png')

segment_height = 40
segment_width = 62

# Esto lo depende del integrante
start_row, start_col = 100, 150

end_row = start_row + segment_height
end_col = start_col + segment_width

image_segment = cv2.cvtColor(image[start_row:end_row, start_col:end_col], cv2.COLOR_BGR2GRAY)

# image_segment = np.array([
#     [232, 224, 131, 232, 233],
#     [215, 231, 234, 230, 118],
#     [230, 218, 226, 219, 226],
#     [215, 106, 228, 230, 232],
#     [224, 221, 234, 226, 217],
# ])

save_channel_to_csv(image_segment, 'segmento_original.csv', fmt='%d')

median = np.median(image_segment)
regional_intensity_mean = np.mean(image_segment)
mode = stats.mode(image_segment, axis=None).mode

print(f'Median: {median}')
print(f'Mean: {regional_intensity_mean}')
print(f'Mode: {mode}')

# Viendo el histograma se debe de definir un umbral optimo para detectar las anomalías
# threshold = 10
threshold = 75

percentual_deviations = np.abs((image_segment - regional_intensity_mean) * 100 / regional_intensity_mean)
print(f'Percentual deviations of original image: {percentual_deviations}', '\n')

counts, bins, patches = plt.hist(percentual_deviations.ravel(), bins=50, color='blue', alpha=0.7)
plt.xlim([50, 100])
plt.title('Histograma de Desviaciones Porcentuales')
plt.xlabel('Desviación Porcentual')
plt.ylabel('Frecuencia')
plt.savefig(f'{histogram_output_folder}/segment_deviations.png')

anomalies = percentual_deviations > threshold

anomalies_rgb = np.zeros((*anomalies.shape, 3), dtype=np.uint8)
anomalies_rgb[anomalies] = [0, 0, 255]  # Rojo
anomalies_rgb[~anomalies] = [0, 0, 0]   # Negro

cv2.imshow('Anomalies', anomalies_rgb)

median_filtered_segment = np.copy(image_segment)
median_filtered_segment[anomalies] = median
median_filtered_regional_intensity_mean = np.mean(median_filtered_segment)
median_filtered_percentual_deviations = np.abs((median_filtered_segment - median_filtered_regional_intensity_mean) * 100 / median_filtered_regional_intensity_mean)
print(f'Intensity mean of regional median filtered image: {median_filtered_regional_intensity_mean}', '\n')
print(f'Percentual deviations of median filtered image: {median_filtered_percentual_deviations}', '\n')
save_channel_to_csv(median_filtered_segment, 'segmento_filtrado_mediana.csv', fmt='%d')

mode_filtered_segment = np.copy(image_segment)
mode_filtered_segment[anomalies] = mode
mode_filtered_regional_intensity_mean = np.mean(mode_filtered_segment)
mode_filtered_percentual_deviations = np.abs((mode_filtered_segment - mode_filtered_regional_intensity_mean) * 100 / mode_filtered_regional_intensity_mean)
save_channel_to_csv(mode_filtered_segment, 'segmento_filtrado_moda.csv', fmt='%d')

print(f'Intensity mean of regional mode filtered image: {mode_filtered_regional_intensity_mean}', '\n')
print(f'Percentual deviations of mode filtered image: {mode_filtered_percentual_deviations}', '\n')

cv2.imshow('Original segment', image_segment.astype(np.uint8))
cv2.imshow('Median filtered segment', median_filtered_segment.astype(np.uint8))
cv2.imshow('Mode filtered segment', mode_filtered_segment.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()