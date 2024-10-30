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
        # print("Neighborhood done")
        return
    
    for i in to_review:
        recursive_run(n_obj[i][0], i, n_obj, n_n, ref_pixel_intensity, ref_pixel_coords)
        # print(i)

    if(c_px_coords == ref_pixel_coords):
        print("NOW Neighborhood done")
        n_n+=1

        new_obj = {
            coords: (intensity, 0 if n_n == -1 else n_n)
            for coords, (intensity, n_n) in n_obj.items()
        }

        new_filtered_keys = [key for key in new_obj.keys() if new_obj[key][1] == 0]
        new_to_review = sorted(new_filtered_keys, key=lambda k: (k[0], k[1]))

        print(new_to_review[0])
        ref = new_obj[new_to_review[0]][0]

        # ref_obj = {
        #     coords: (intensity, n_n)
        #     for coords, (intensity, n_n) in new_obj.items()
        #     if coords == new_to_review[0]
        # }

        n_obj[new_to_review[0]] =  (ref, 0)

        recursive_run(ref, new_to_review[0], n_obj, n_n, ref_pixel_intensity=ref, ref_pixel_coords=new_to_review[0])

        # recursive_run(c_px_i, c_px_coords, n_obj, -1)
        # for i in new_to_review:
        #     ref = new_obj[new_to_review[0]][0]
        
            # recursive_run(new_obj[i][0], i, new_obj, n_n, ref_pixel_intensity=ref)
        
        # print(i)

        # print(ref_obj)
        print()
        

    pass

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



run(intensity_matrix)