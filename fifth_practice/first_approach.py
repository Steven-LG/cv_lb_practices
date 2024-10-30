
import numpy as np


intensity_matrix = np.array(
    [
        [138, 132, 123, 145, 147],
        [145, 122, 131, 135, 126],
        [150, 147, 138, 123, 145],
        [137, 128, 123, 122, 122],
        [130, 124, 120, 142, 135]
    ]
)

def get_neighbors(matrix, row, col):
    rows, cols = matrix.shape
    neighbors = []

    for i in range(row-1, row+2):
        for j in range(col-1, col+2):
            if (0 <= i < rows) and (0 <= j < cols) and (i != row or j != col):
                neighbors.append((i, j))

    return neighbors

def create_neighboorhood(matrix):
    intensity_difference = 10

    reference_pixel = intensity_matrix[0][0]
    reference_neighbors_coords = get_neighbors(matrix=matrix, row=0, col=0)
    
    reference_neighbors = {coord: matrix[coord] for coord in reference_neighbors_coords}

    filtered_neighbors = {
        coords: value
        for coords, value in reference_neighbors.items()
        if (reference_pixel - intensity_difference) < value < (reference_pixel + intensity_difference)
    }

    same_row_neighbors = {
        coords: value
        for coords, value in filtered_neighbors.items()
        if coords[0] == 0
    }   

    # traverse the x axis from the filtered neighbors

    for coords, value in same_row_neighbors.items():
        new_neighbors = get_neighbors(matrix, coords[0], coords[1])
        new_reference_neighbors = {coord: matrix[coord] for coord in new_neighbors}

        all_neighbors = {**reference_neighbors, **new_reference_neighbors}


    # neighbor_candidates = ((reference_pixel-intensity_difference) < matrix) & (matrix < (reference_pixel+intensity_difference))

    # Get the coordinates of the pixel candidates
    # candidates_coordinates = np.where(neighbor_candidates)

    # new_neighboorhood_dict = {}
    # for coords in zip(candidates_coordinates[0], candidates_coordinates[1]):
    #     if coords == (0, 0):
    #         new_neighboorhood_dict[coords] = reference_pixel
    #         continue

    #     # In the third iteration the current value is 123, so it is not in the reference neighbors
    #     current = neighbor_candidates[coords]
    #     new_neighboorhood_dict[coords] = int(current)
    
    # desired_row = 0
    # candidates = [intensity for coords, intensity in new_neighboorhood_dict.items() if coords[0] == desired_row]

    # print(candidates)
    print()
    # to_review = []

# intensity_difference = int(input("Cual es la diferencia de intensidad?"))



create_neighboorhood(intensity_matrix)
