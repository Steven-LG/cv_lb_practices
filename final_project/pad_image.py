import os
from PIL import Image, ImageOps

input_dir = '/home/xsvd/Main/Homework/LBVC/final_project/dataset/healthy'
output_dir = '/home/xsvd/Main/Homework/LBVC/final_project/new_dataset/healthy'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Process only the first 50 images
for filename in image_files[:50]:
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)

    # Resize the image to 352x352
    resized_img = img.resize((352, 352), Image.LANCZOS)

    # Add 257 pixels of black padding to top and bottom
    padding = (0, 257, 0, 257)  # (left, top, right, bottom)
    padded_img = ImageOps.expand(resized_img, padding, fill='black')

    output_path = os.path.join(output_dir, filename)
    padded_img.save(output_path)