
import numpy as np
import cv2

intensity_matrix = np.array(
    [
        [138, 132, 123, 145, 147],
        [145, 122, 131, 135, 126],
        [150, 147, 138, 123, 145],
        [137, 128, 123, 122, 122],
        [130, 124, 120, 142, 135]
    ]
)
reference_pixel_intensity = np.int64(intensity_matrix[0][0])
intensity_difference = 10


def get_neighbors_coords(matrix, x, y):
    rows, cols = matrix.shape
    neighbors = []

    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if (0 <= i < rows) and (0 <= j < cols) and (i != x or j != y):
                neighbors.append((i, j))

    return neighbors

def is_px_within_range(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity=reference_pixel_intensity):
    if (ref_pixel_intensity - intensity_difference) <= c_px_i <= (ref_pixel_intensity + intensity_difference):
        if not(c_px_coords in n_obj):
            n_obj[c_px_coords] = (c_px_i, n_n)
        if n_obj[c_px_coords][1] == 0:
            n_obj[c_px_coords] = (c_px_i, n_n)
        return True
    
    n_obj[c_px_coords] = (c_px_i, -1)
    return False

def recursive_run(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity=reference_pixel_intensity):
    if not(is_px_within_range(c_px_i, c_px_coords, n_obj, n_n, ref_pixel_intensity=ref_pixel_intensity)):
        return
    
    neighbors_coords = get_neighbors_coords(intensity_matrix, x=c_px_coords[0], y=c_px_coords[1])
    
    for coords in neighbors_coords:
        if not(coords in n_obj):
            n_obj[coords] = (intensity_matrix[coords], 0)
    
    filtered_keys = [key for key in n_obj.keys() if n_obj[key][1] == 0]
    to_review = sorted(filtered_keys, key=lambda k: (k[0], k[1]))

    if len(to_review) == 0:
        # print("Neighborhood done")
        return
    
    for i in to_review:
        recursive_run(n_obj[i][0], i, n_obj, n_n)
        # print(i)

    if(c_px_coords == (0, 0)):
        print("NOW Neighborhood done")
        n_n+=1

        # print(n_obj)

        new_obj = {
            coords: (intensity, 0 if n_n == -1 else n_n)
            for coords, (intensity, n_n) in n_obj.items()
        }

        # print(new_obj)

        new_filtered_keys = [key for key in new_obj.keys() if new_obj[key][1] == 0]
        new_to_review = sorted(new_filtered_keys, key=lambda k: (k[0], k[1]))

        # recursive_run(c_px_i, c_px_coords, n_obj, -1)
        for i in new_to_review:
            ref = new_obj[new_to_review[0]][0]
            recursive_run(new_obj[i][0], i, new_obj, n_n, ref_pixel_intensity=ref)
        
        # print(i)

        print(new_obj)
        print()
        # rows, cols = 5, 5  # Size of the original matrix
        # image = np.zeros((rows, cols), dtype=np.uint8)

        # for (x, y), (_, n_n) in n_obj.items():
        #     if n_n == 1:
        #         image[x, y] = 0
        #     if n_n == -1:
        #         image[x, y] = 255

        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    pass

def run(matrix):
    r_px_i = matrix[0][0]
    n_obj = {}
    r_px_coords = (0, 0)

    recursive_run(r_px_i, r_px_coords, n_obj, 1)
    
    print()



run(intensity_matrix)