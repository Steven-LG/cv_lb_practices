import numpy as np
import cv2
import matplotlib.pyplot as plt

intensity_matrix = np.array(
    [
        [138, 132, 123, 145, 147],
        [145, 122, 131, 135, 126],
        [150, 147, 138, 123, 145],
        [137, 128, 123, 122, 122],
        [130, 124, 120, 142, 135]
    ]
)

intensity_difference = 10

def get_neighbors_coords(matrix, x, y):
    rows, cols = matrix.shape
    neighbors = []

    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if (0 <= i < rows) and (0 <= j < cols) and (i != x or j != y):
                neighbors.append((i, j))

    return neighbors

def is_px_within_range(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity):
    if (ref_pixel_intensity - intensity_difference) <= c_px_i <= (ref_pixel_intensity + intensity_difference):
        if not(c_px_coords in n_obj):
            n_obj[c_px_coords] = (c_px_i, n_n)
        if n_obj[c_px_coords][1] == 0:
            n_obj[c_px_coords] = (c_px_i, n_n)
        return True
    
    n_obj[c_px_coords] = (c_px_i, -1)
    return False
# Function to get color based on intensity
def get_color(intensity, thresholds, colors):
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= intensity < thresholds[i + 1]:
            return colors[i]
    return colors[-1]  # Assign the last color if intensity equals max_intensity

def generate_color_map(unique_values):
    cmap = plt.get_cmap('tab10', len(unique_values))  # 'tab10' has 10 distinct colors
    color_map = {}
    for idx, value in enumerate(unique_values):
        color = cmap(idx)[:3]  # Get RGB values (ignore alpha)
        color = tuple(int(255 * c) for c in color)  # Scale to 0-255
        color_map[value] = color
    return color_map


def recursive_run(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity, ref_pixel_coords):
    if not(is_px_within_range(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity=ref_pixel_intensity)):
        return
    
    neighbors_coords = get_neighbors_coords(intensity_matrix, x=c_px_coords[0], y=c_px_coords[1])
    
    for coords in neighbors_coords:
        if not(coords in n_obj):
            n_obj[coords] = (intensity_matrix[coords], 0)
        if n_obj[coords][1] == -1:
            n_obj[coords] = (intensity_matrix[coords], 0)
        
    
    filtered_keys = [key for key in n_obj.keys() if n_obj[key][1] == 0]
    to_review = sorted(filtered_keys, key=lambda k: (k[0], k[1]))

    if len(to_review) == 0:
        return
    
    for i in to_review:
        recursive_run(n_obj[i][0], i, n_obj, n_n, ref_pixel_intensity, ref_pixel_coords)

    if(c_px_coords == ref_pixel_coords):
        n_n+=1

        new_obj = {
            coords: (intensity, 0 if n_n == -1 else n_n)
            for coords, (intensity, n_n) in n_obj.items()
        }

        new_filtered_keys = [key for key in new_obj.keys() if new_obj[key][1] == 0]
        new_to_review = sorted(new_filtered_keys, key=lambda k: (k[0], k[1]))

        ref = new_obj[new_to_review[0]][0]

        n_obj[new_to_review[0]] =  (ref, 0)

        recursive_run(ref, new_to_review[0], n_obj, n_n, ref_pixel_intensity=ref, ref_pixel_coords=new_to_review[0])

def iterative_run_2(matrix, start_px_coords, n_obj, n_n, ref_pixel_intensity):
    stack = [start_px_coords]
    while stack:
        c_px_coords = stack.pop()
        if c_px_coords in n_obj:
            continue
        c_px_i = matrix[c_px_coords]
        if not is_px_within_range(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity):
            continue
        n_obj[c_px_coords] = (c_px_i, n_n)
        neighbors_coords = get_neighbors_coords(matrix, x=c_px_coords[0], y=c_px_coords[1])
        for coords in neighbors_coords:
            if coords not in n_obj:
                stack.append(coords)

def iterative_run_1(start_px_coords, n_obj, n_n, ref_pixel_intensity):
    stack = [start_px_coords]

    while stack:
        c_px_coords = stack.pop()
        c_px_i = intensity_matrix[c_px_coords]

        if not is_px_within_range(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity):
            continue
        
        neighbors_coords = get_neighbors_coords(intensity_matrix, x=c_px_coords[0], y=c_px_coords[1])
        for coords in neighbors_coords:
            if coords not in n_obj or n_obj[coords][1] == -1:
                n_obj[coords] = (intensity_matrix[coords], 0)
                stack.append(coords)

def run(matrix):
    r_px_i = matrix[0][0]
    n_obj = {}
    r_px_coords = (0, 0)

    recursive_run(r_px_i, r_px_coords, n_obj, 1, ref_pixel_intensity=r_px_i, ref_pixel_coords=r_px_coords)

    # Size of the original matrix
    rows, cols = 5, 5  
    image = np.zeros((rows, cols), dtype=np.uint8)

    unique_n_n = sorted(set(n_n for _, n_n in n_obj.values()))

    n_n_to_color = generate_color_map(unique_n_n)

    # Determine the size of the image
    max_x = max(x for x, _ in n_obj.keys()) + 1
    max_y = max(y for _, y in n_obj.keys()) + 1

    # Create a blank RGB image
    image = np.zeros((max_x, max_y, 3), dtype=np.uint8)

    # Assign colors to the image based on n_n values
    for (x, y), (_, n_n) in n_obj.items():
        color = n_n_to_color.get(n_n, (0, 0, 0))  # Default to black if n_n not found
        image[x, y] = color

    cv2.imshow('n_n Colored Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_i_1(matrix):
    n_obj = {}
    n_n = 1  # Initialize the group counter or label
    start_px_coords = (0, 0)
    ref_pixel_intensity = matrix[start_px_coords]

    # Call the iterative function instead of the recursive one
    iterative_run_1(start_px_coords, n_obj, n_n, ref_pixel_intensity)


    print("Iterative run done.")

    # Proceed with the rest of the code to visualize or process n_obj
    unique_n_n = sorted(set(n_n for _, n_n in n_obj.values()))
    n_n_to_color = generate_color_map(unique_n_n)
    
    # Determine the size of the image
    max_x = max(x for x, _ in n_obj.keys()) + 1
    max_y = max(y for _, y in n_obj.keys()) + 1

    # Create a blank RGB image
    image = np.zeros((max_x, max_y, 3), dtype=np.uint8)

    # Assign colors to the image based on n_n values
    for (x, y), (_, n_n) in n_obj.items():
        color = n_n_to_color.get(n_n, (0, 0, 0))  # Default to black if n_n not found
        image[x, y] = color

    cv2.imshow('n_n Colored Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_i_2(matrix):
    n_obj = {}
    n_n = 1  # Initialize the group counter
    rows, cols = matrix.shape
    total_pixels = rows * cols

    while len(n_obj) < total_pixels:
        # Find an unprocessed pixel
        unprocessed_pixels = [(x, y) for x in range(rows) for y in range(cols) if (x, y) not in n_obj]
        if not unprocessed_pixels:
            break
        start_px_coords = unprocessed_pixels[0]
        ref_pixel_intensity = matrix[start_px_coords]
        
        # Run the iterative function for the current group
        iterative_run_2(matrix, start_px_coords, n_obj, n_n, ref_pixel_intensity)
        
        # Increment n_n for the next group
        n_n += 1

    # Proceed with visualization
    unique_n_n = sorted(set(n for _, n in n_obj.values()))
    n_n_to_color = generate_color_map(unique_n_n)
    
    # Determine the size of the image
    max_x = rows
    max_y = cols

    # Create a blank RGB image
    image = np.zeros((max_x, max_y, 3), dtype=np.uint8)

    # Assign colors to the image based on n_n values
    for (x, y), (_, n_n_value) in n_obj.items():
        color = n_n_to_color.get(n_n_value, (0, 0, 0))  # Default to black if n_n not found
        image[x, y] = color

    cv2.imshow('n_n Colored Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run_i_2(intensity_matrix)