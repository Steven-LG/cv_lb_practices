import numpy as np

def get_neighbors(matrix, row, col):
    rows, cols = matrix.shape
    neighbors = []

    for i in range(row-1, row+2):
        for j in range(col-1, col+2):
            if (0 <= i < rows) and (0 <= j < cols) and (i != row or j != col):
                neighbors.append((i, j))

    return neighbors

def recursive_run(value, coords, n_obj, n_n):
    # Base case: Check if there are no more neighbors to review
    to_review = sorted([key for key in n_obj.keys() if n_obj[key][1] == 0], key=lambda k: (k[0], k[1]))
    if len(to_review) == 0:
        print("Neighborhood done")
        return
    
    # Debugging: Print the current state
    print(f"Running recursion for coords: {coords}, value: {value}")
    
    # Recursive case: Ensure that the recursion progresses towards the base case
    new_neighbors = get_neighbors(n_obj, coords[0], coords[1])
    for neighbor in new_neighbors:
        if neighbor not in n_n:
            n_n[neighbor] = n_obj[neighbor]
            recursive_run(n_obj[neighbor][0], neighbor, n_obj, n_n)

# Example usage
n_obj = {
    (0, 0): (np.int64(138), 1),
    (0, 1): (np.int64(132), 0),
    (1, 0): (np.int64(145), 0),
    (1, 1): (np.int64(122), 0)
}

to_review = sorted([key for key in n_obj.keys() if n_obj[key][1] == 0], key=lambda k: (k[0], k[1]))

if len(to_review) == 0:
    print("Neighborhood done")
else:
    n_n = {}
    for i in to_review:
        recursive_run(n_obj[i][0], i, n_obj, n_n)
        # print(i)