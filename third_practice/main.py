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

start_row, start_col = 100, 150

end_row = start_row + segment_height
end_col = start_col + segment_width

image_segment = cv2.cvtColor(image[start_row:end_row, start_col:end_col], cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Segment', cv2.resize(image_segment, (segment_width * 10, segment_height * 10), interpolation=cv2.INTER_NEAREST))
save_channel_to_csv(image_segment, 'segmento_original.csv', fmt='%d')

print(np.array(image_segment))

median = np.median(image_segment)
print(f'Median: {median} \n')
mean, std_dev = cv2.meanStdDev(image_segment)
mean = mean[0][0]
threshold = 1.5
print(f'Mean: {mean} \n')

mode_value = stats.mode(image_segment, axis=None).mode
print(f'Mode: {mode_value} \n')

deviations_percent = np.abs((mean - image_segment * 100) / mean)

percentile_threshold = np.percentile(deviations_percent, 95)
print(f'Percentil 95: {percentile_threshold}')

print("Matriz de desviaciones:")
print(deviations_percent)
save_channel_to_csv(deviations_percent, 'desviaciones.csv')

anomalies = deviations_percent > threshold
print("Matriz de anomalias:")
print(anomalies)
save_channel_to_csv(anomalies, 'anomalias.csv', fmt='%d')

# use_mean_filter = 1
filtered_segment = np.copy(image_segment)
for i in range(image_segment.shape[0]):
    for j in range(image_segment.shape[1]):
        if filtered_segment[i, j] > percentile_threshold:
            # if use_mean_filter == 1:
            filtered_segment[i, j] = median
            # else:
            #     filtered_segment[i, j] = mode_value


# filtered_segment[anomalies] = median

print(filtered_segment)
cv2.imshow('Image with filtered segment', cv2.resize(filtered_segment, (segment_width * 10, segment_height * 10), interpolation=cv2.INTER_NEAREST))

counts, bins, patches = plt.hist(deviations_percent.ravel(), bins=50, color='blue', alpha=0.7)
# plt.xticks(bins, rotation=90)
plt.title('Histograma de Desviaciones Porcentuales')
plt.xlabel('Desviación Porcentual')
plt.ylabel('Frecuencia')
plt.savefig(f'{histogram_output_folder}/segment_deviations.png')

# plt.hist(image_segment.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
# plt.title('Histograma de intensidades de píxeles')
# plt.xlabel('Intensidad de píxeles')
# plt.ylabel('Frecuencia')
# plt.savefig(f'{histogram_output_folder}/segment_intentisies.png')

cv2.waitKey(0)
cv2.destroyAllWindows()

