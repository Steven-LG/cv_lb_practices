import os 
import numpy as np
import cv2

img = cv2.imread('first_practice/incendio.png',)

b,g,r = cv2.split(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

output_folder = 'first_practice/csv_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def save_channel_to_csv(channel, filename):
    filepath = os.path.join(output_folder, filename)
    np.savetxt(filepath, channel, delimiter=',', fmt='%d')

save_channel_to_csv(r, 'canal_rojo.csv')
save_channel_to_csv(g, 'canal_verde.csv')
save_channel_to_csv(b, 'canal_azul.csv')
save_channel_to_csv(gray, 'escala_grises.csv')

converted_output_folder = 'first_practice/converted_image_output'
if not os.path.exists(converted_output_folder):
    os.makedirs(converted_output_folder)

red_image_path = os.path.join(converted_output_folder, 'imagen_rojo.png')
green_image_path = os.path.join(converted_output_folder, 'imagen_verde.png')
blue_image_path = os.path.join(converted_output_folder, 'imagen_azul.png')
gray_image_path = os.path.join(converted_output_folder, 'imagen_gris.png')

cv2.imwrite(red_image_path, r)
cv2.imwrite(green_image_path, g)
cv2.imwrite(blue_image_path, b)
cv2.imwrite(gray_image_path, gray)

