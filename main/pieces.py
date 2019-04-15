import numpy as np

# each piece is a dictionary of `data`, `neighbors`, `diagonals` and `rotflip`
pieces = {}
num_pieces = 21
with open('pieces_py.txt', 'r') as f:
    for idx, line in enumerate(f):
        block, corners, neighbors_idx, diagonals_idx, meta = eval(line)

        block = np.array(block, dtype=np.int8)
        corners = np.array(corners, dtype=np.int8)
        width, height = len(block[0]), len(block)
        neighbors = np.zeros((height + 2, width + 2), dtype=np.int8)
        diagonals = np.zeros((height + 2, width + 2), dtype=np.int8)
        for _i, _j in neighbors_idx:
            neighbors[1+_i, 1+_j] = 1
        for _i, _j in diagonals_idx:
            diagonals[1+_i, 1+_j] = 1

        piece = {}
        for i in range(4):
            rot_block = np.rot90(block, i)
            rot_corners = np.rot90(corners, i)
            rot_neighbors = np.rot90(neighbors, i)
            rot_diagonals = np.rot90(diagonals, i)

            rot_already_in = np.array([np.array_equal(rot_block, d)
                                       for d in [d[0] for d in piece.values()]]).any()
            if not rot_already_in:
                piece[(i, 0)] = [rot_block, rot_corners,
                                 rot_neighbors, rot_diagonals]

            flip_block = np.fliplr(rot_block)
            flip_corners = np.fliplr(rot_corners)
            flip_neighbors = np.fliplr(rot_neighbors)
            flip_diagonals = np.fliplr(rot_diagonals)

            flip_already_in = np.array([np.array_equal(flip_block, d)
                                        for d in [d[0] for d in piece.values()]]).any()
            if not flip_already_in:
                piece[(i, 1)] = [flip_block, flip_corners,
                                 flip_neighbors, flip_diagonals]
        pieces[idx] = piece

num_pieces = len(pieces)