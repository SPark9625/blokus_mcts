import numpy as np
from copy import deepcopy
from collections import namedtuple

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

State = namedtuple('State', ['board', 'remaining_pieces_all', 'player', 'first', 'done', 'meta'])
Meta = namedtuple('Meta', ['actions', 'new_diagonals'])

class Blokus:
  def __init__(self, size, player_list):
    self.size = size
    self.player_list = player_list
    self.num_players = len(player_list)

    # each piece is a dictionary of `data`, `neighbors`, `diagonals` and `rotflip`
    self.pieces = {}
    with open('pieces.txt', 'r') as f:
      for idx, line in enumerate(f):
        block, corners, neighbors_idx, diagonals_idx, meta = eval(line)

        block = np.array(block, dtype=np.uint8)
        corners = np.array(corners, dtype=np.uint8)
        width, height = len(block[0]), len(block)
        neighbors = np.zeros((height + 2, width + 2), dtype=np.uint8)
        diagonals = np.zeros((height + 2, width + 2), dtype=np.uint8)
        for _i,_j in neighbors_idx:
          neighbors[1+_i,1+_j] = 1
        for _i,_j in diagonals_idx:
          diagonals[1+_i,1+_j] = 1

        piece = {}
        for i in range(4):
          rot_block = np.rot90(block, i)
          rot_corners = np.rot90(corners, i)
          rot_neighbors = np.rot90(neighbors, i)
          rot_diagonals = np.rot90(diagonals, i)

          rot_already_in = np.array([np.array_equal(rot_block, d) for d in [d[0] for d in piece.values()]]).any()
          if not rot_already_in:
            piece[(i,0)] = [rot_block, rot_corners, rot_neighbors, rot_diagonals]

          flip_block = np.fliplr(rot_block)
          flip_corners = np.fliplr(rot_corners)
          flip_neighbors = np.fliplr(rot_neighbors)
          flip_diagonals = np.fliplr(rot_diagonals)

          flip_already_in = np.array([np.array_equal(flip_block, d) for d in [d[0] for d in piece.values()]]).any()
          if not flip_already_in:
            piece[(i,1)] = [flip_block, flip_corners, flip_neighbors, flip_diagonals]
        self.pieces[idx] = piece

  def reset(self):
    '''board[0]: main board,
    board[1 : num_players + 1]: diagonals,
    board[num_players + 1 : ]: neighbors
    player = int(1 ~ num_players)
    player_list = list of players
    remaining_pieces_all = dict with key(player), value(list of available piece indices)
    first = dict with key(player), value(bool)
    '''
    layer_depth = 1 + 2 * self.num_players
    board = np.zeros((layer_depth, self.size, self.size), dtype=np.uint8)

    # # for all players, they can place their piece in any corner on the board on their first move
    # board[1:self.num_players+1, [0,0,self.size-1,self.size-1],[0,self.size-1,0,self.size-1]] = 1

    # for now, just let the two players place their piece on either the topleft corner or the bottom right corner
    # This assumes TWO players
    first_pos = np.array([[[0,0]],[[self.size-1,self.size-1]]])
    for p in range(self.num_players):
      board[1 + p][tuple(first_pos[p,0])] = 1

    piece_keys = [list(self.pieces.keys()) for i in range(self.num_players)]
    remaining_pieces_all = {p: p_list for p, p_list in zip(self.player_list, piece_keys)}
    player = self.player_list[0]
    first = {p: True for p in self.player_list}
    done = {p: False for p in self.player_list}

    new_diagonals_all = {p:first_pos[i] for i, p in enumerate(self.player_list)}
    actions_all = {p:[] for p in self.player_list}
    meta = Meta(actions_all, new_diagonals_all)

    state = State(board, remaining_pieces_all, player, first, done, meta)
    
    return state


  def step(self, state, action, actions):
    # player_list and remaining_pieces_all required in order to check if game has ended.
    # player = 1 ~ 4
    # action = (piece_id, i, j, rotation, flip)
    # this assumes a valid action
    next_board = state.board.copy()
    remaining_pieces_all = deepcopy(state.remaining_pieces_all)
    first = deepcopy(state.first)
    done = deepcopy(state.done)
    player = state.player

    can_place = self.place_possible(state.board, player, action)
    if not can_place:
      raise ValueError(f"You can't place here.\nplayer: {player}\naction: {action}\nboard:\n{state.board}")

    idx, i, j, rot, flip = action
    
    # later change back
    try:
      block, corners, neighbors, diagonals = deepcopy(self.pieces[idx][(rot, flip)])
    except:
      block, corners, neighbors, diagonals = self.__adjust(self.pieces[idx], rot, flip)

    block *= player
    width, height = len(block[0]), len(block)
    # update main board: assumes a valid action, thus just add.
    next_board[0, i:i+height, j:j+width] += block

    if first[player]:
      next_board[player,[0,0,self.size-1,self.size-1],[0,self.size-1,0,self.size-1]] = 0
      first[player] = False

    # calculate outer slices
    x_left, y_top = j-1,i-1
    x_right, y_bottom = j+width+1, i+height+1
    diagonals, neighbors, slice_meta = self.fit_to_board(x_left, y_top, x_right, y_bottom, diagonals, neighbors)
    x_slice, y_slice = slice_meta
    
    # update neighbors
    existing_neighbors = next_board[self.num_players + player, y_slice, x_slice]
    neighbors[np.logical_and(existing_neighbors, neighbors)] = 0
    existing_neighbors += neighbors

    # update diagonals -- this has to be perfectly accurate at all times.
    # firstly remove any existing diagonals where a piece is about to be placed.
    focus = next_board[1:1+self.num_players, i:i+height, j:j+width]
    focus[np.logical_and(focus, block)] = 0

    outer_diagonals = next_board[player, y_slice, x_slice]
    outer_diagonals[np.logical_and(outer_diagonals, neighbors)] = 0    

    # If places where the new piece is about to declare diagonal are already placed, they can't be.
    main_board = next_board[0, y_slice, x_slice]
    diagonals[np.logical_and(main_board, diagonals)] = 0

    # same with places adjacent to the player's own color
    neighbor_board = next_board[self.num_players+player, y_slice, x_slice]
    diagonals[np.logical_and(neighbor_board, diagonals)] = 0

    # then add diagonals
    diagonal_focus = next_board[player, y_slice, x_slice]
    diagonals[np.logical_and(diagonal_focus, diagonals)] = 0
    next_board[player, y_slice, x_slice] += diagonals
    new_diagonals = np.argwhere(diagonals) + [y_slice.start, x_slice.start]

    actions_all = deepcopy(state.meta.actions)
    new_diagonals_all = deepcopy(state.meta.new_diagonals)

    actions = [a for a in actions if a[0] != action[0]]
    actions_all[state.player] = actions
    new_diagonals_all[state.player] = new_diagonals
    meta = Meta(actions_all, new_diagonals_all)
    
    remaining_pieces_all[player].remove(action[0])
    
    game_done = self._check_game_finished(next_board, remaining_pieces_all, done, meta)
    next_player = self.next_player(player, done)

    next_state = State(next_board, remaining_pieces_all, next_player, first, done, meta)

    reward = {p:0 for p in self.player_list}
    if game_done:
      scores = []
      for idx in self.player_list:
        score = np.sum(next_board[0] == idx)
        scores.append(score)
      if scores[0] > scores[1]:
        reward = {1:1, 2:-1}
      elif scores[0] < scores[1]:
        reward = {1:-1, 2:1}
    return next_state, reward, game_done, None

  def possible_actions(self, state, player):
    # Use metadata s.t. only disabled actions are removed and additional diagonals are checked
    prev_actions = state.meta.actions[player]

    i_s,j_s = state.meta.new_diagonals[player].T

    dead_actions = set()
    for i, action in enumerate(prev_actions):
      if not self.place_possible(state.board, player, action) or action[0] not in state.remaining_pieces_all[player]:
        dead_actions.add(action)
    prev_actions = list(set(prev_actions) - dead_actions)
    
    alive = state.board[player, i_s, j_s].astype(bool)
    new_diagonals = state.meta.new_diagonals[player][alive]

    new_actions = []
    for idx in state.remaining_pieces_all[player]:
      piece = self.pieces[idx]
      for (r, f), data in piece.items():
        block, corners, neighbors, diagonals = data
        width, height = len(block[0]), len(block)
        for diag_pos in new_diagonals:
          for offset in np.argwhere(corners):
            pos = diag_pos - offset
            # if coord goes off the board, ignore and continue
            if np.any(pos < 0) or np.any(pos+[height, width] > self.size):
              continue
            else:
              i, j = pos
              action = (idx, i, j, r, f)
              if self.place_possible(state.board, player, action):
                new_actions.append(action)
    actions = prev_actions + new_actions
    return actions

  def place_possible(self, board, player, action):
    idx, i, j, rot, flip = action

    # later change back
    piece = self.pieces[idx]
    try:
      block, corners, neighbors, diagonals = piece[(rot, flip)]
    except:
      block, corners, neighbors, diagonals = self.__adjust(piece, rot, flip)
    width, height = len(block[0]), len(block)

    # check overlap
    focus = board[0, i:i+height, j:j+width]
    if np.any(np.logical_and(block, focus)):
      return False

    # check if there are any common flat edge
    focus = board[self.num_players + player, i:i+height, j:j+width]
    if np.any(np.logical_and(block, focus)):
      return False

    # make sure the corners touch
    focus = board[player, i:i+height, j:j+width]
    if np.any(np.logical_and(block, focus)):
      return True

    return False

  def _check_player_finished(self, board, remaining_pieces_all, player, meta):
    prev_actions = meta.actions[player]
    i_s,j_s = meta.new_diagonals[player].T

    for i, action in enumerate(prev_actions):
      if self.place_possible(board, player, action):
        return False
    
    alive = board[player, i_s, j_s].astype(bool)
    new_diagonals = meta.new_diagonals[player][alive]
    for idx in remaining_pieces_all[player]:
      piece = self.pieces[idx]
      for (r, f), data in piece.items():
        block, corners, neighbors, diagonals = data
        width, height = len(block[0]), len(block)
        for diag_pos in new_diagonals:
          for offset in np.argwhere(corners):
            pos = diag_pos - offset
            # if coord goes off the board, ignore and continue
            if np.any(pos < 0) or np.any(pos+[height, width] > self.size):
              continue
            else:
              i, j = pos
              action = (idx, i, j, r, f)
              if self.place_possible(board, player, action):
                return False
    return True

  def _check_game_finished(self, board, remaining_pieces_all, done, meta):
    finished = True
    for player in self.player_list:
      if not done[player]:
        if self._check_player_finished(board, remaining_pieces_all, player, meta):
          done[player] = True
        else:
          finished = False
    return finished

  @staticmethod
  def __adjust(piece, r, f):
    block = piece[(0,0)][0].copy()
    corners = piece[(0,0)][1].copy()
    neighbors = piece[(0,0)][2].copy()
    diagonals = piece[(0,0)][3].copy()

    block = np.rot90(block, r)
    corners = np.rot90(corners, r)
    neighbors = np.rot90(neighbors, r)
    diagonals = np.rot90(diagonals, r)

    if f:
      block = np.fliplr(block)
      corners = np.fliplr(corners)
      neighbors = np.fliplr(neighbors)
      diagonals = np.fliplr(diagonals)

    return block, corners, neighbors, diagonals


  def fit_to_board(self, x_left, y_top, x_right, y_bottom, diagonals, neighbors):
    if x_left < 0:
      diff = -x_left
      x_left = 0
      neighbors = neighbors[:, diff:]
      diagonals = diagonals[:, diff:]

    if x_right > self.size:
      diff = x_right - self.size
      x_right = self.size
      neighbors = neighbors[:, :-diff]
      diagonals = diagonals[:, :-diff]
      
    if y_top < 0:
      diff = -y_top
      y_top = 0
      neighbors = neighbors[diff:]
      diagonals = diagonals[diff:]
    if y_bottom > self.size:
      diff = y_bottom - self.size
      y_bottom = self.size
      neighbors = neighbors[:-diff]
      diagonals = diagonals[:-diff]
    x_slice = slice(x_left, x_right)
    y_slice = slice(y_top, y_bottom)
    return diagonals, neighbors, (x_slice, y_slice)

  def next_player(self, player, done):
    """
    This assumes that players are ordered integers.
    If no one is alive, returns None
    """
    alive = [p for p, dead in done.items() if not dead]
    if not alive:
      return None
    for p in alive:
      if p > player:
        return p
    return alive[0]