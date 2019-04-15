from pieces import pieces, num_pieces

# `layer2irf` maps action layer id to (idx, r, f)
layer2irf = []

# `idx2slice` maps piece idx to slice on the layer2irf
idx2slice = [None for i in range(num_pieces)]
for i in range(num_pieces):
    keys = list(pieces[i].keys())
    start = len(layer2irf)
    end = start + len(keys)

    layer2irf += [(i, r, f) for r, f in keys]
    idx2slice[i] = slice(start, end)

# `irf2layer` maps (idx, r, f) to action layer id
irf2layer = {key: i for i, key in enumerate(layer2irf)}
num_action_layers = len(layer2irf)



class config:
    # ------------------------ YOU CAN MODIFY ------------------------ #
    env_mode = 'py'

    # Has to be 2 or 4
    num_players = 2
    howlongtokeeptau1 = 5

    framework   = 'tensorflow'
    data_format = 'channels_last'
    num_resblock = 1

    c_puct = 3

    num_selectors = 1
    num_workers = 2
    time_budget = 60
    iter_budget = 400

    max_batch_size = 32
    # temp
    print = 1
    # ----------------------------- END ------------------------------ #

    # ------------------------- DO NOT TOUCH ------------------------- #
    layer_idx = 2 if data_format == 'channels_last' else 0
    pieces     = pieces
    num_pieces = num_pieces
    
    layer2irf = layer2irf
    irf2layer = irf2layer
    idx2slice = idx2slice

    player_list = list(range(num_players))

    num_state_layers  = num_players * (num_pieces + 4) + 1
    num_action_layers = num_action_layers

    if num_players == 2:
        board_size = 13
    elif num_players == 4:
        board_size = 20
    else: raise ValueError(f'`num_players` has to be either 2 or 4, but got {num_players}')
    
    board_shape = (num_state_layers, board_size, board_size)
    if data_format == "channels_last":
        action_shape = (board_size, board_size, num_action_layers)
    else:
        action_shape = (num_action_layers, board_size, board_size)

    diagonal = num_players * num_pieces
    neighbor = diagonal + num_players
    first = neighbor + num_players
    done = first + num_players
    turn = -1

    alpha = 10 / board_size ** 2
    # ----------------------------- END ------------------------------ #
