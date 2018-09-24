import numpy as np
from copy import deepcopy
from collections import namedtuple

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

State = namedtuple('State', ['board', 'remaining_pieces_all', 'first', 'done'])


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
        # all_data = set()
        for i in range(4):
          rot_block = np.rot90(block, i)
          rot_corners = np.rot90(corners, i)
          rot_neighbors = np.rot90(neighbors, i)
          rot_diagonals = np.rot90(diagonals, i)

          # block_data = [d['block'] for d in piece.values()]
          rot_already_in = np.array([np.array_equal(rot_block, d) for d in [d[0] for d in piece.values()]]).any()
          # rot_already_in = (rot_block.tobytes() in all_data)
          if not rot_already_in:
            # all_data.add(rot_block.tobytes())
            piece[(i,0)] = [rot_block, rot_corners, rot_neighbors, rot_diagonals]
            # {
            #   'block': rot_block,
            #   'corners': rot_corners,
            #   'neighbors': rot_neighbors,
            #   'diagonals': rot_diagonals
            # }

          flip_block = np.fliplr(rot_block)
          flip_corners = np.fliplr(rot_corners)
          flip_neighbors = np.fliplr(rot_neighbors)
          flip_diagonals = np.fliplr(rot_diagonals)

          flip_already_in = np.array([np.array_equal(flip_block, d) for d in [d[0] for d in piece.values()]]).any()
          # flip_already_in = (flip_block.tobytes() in all_data)
          if not flip_already_in:
            # all_data.add(flip_block.tobytes())
            piece[(i,1)] = [flip_block, flip_corners, flip_neighbors, flip_diagonals]
            # {
            #   'block': flip_block,
            #   'corners': flip_corners,
            #   'neighbors': flip_neighbors,
            #   'diagonals': flip_diagonals
            # }
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
    first_pos = [(0,0),(self.size-1, self.size-1)]
    for p in range(self.num_players):
      board[1 + p][first_pos[p]] = 1

    piece_keys = [list(self.pieces.keys()) for i in range(self.num_players)]
    remaining_pieces_all = {p: p_list for p, p_list in zip(self.player_list, piece_keys)}
    first = {p: True for p in self.player_list}
    done = {p: False for p in self.player_list}
    
    state = State(board, remaining_pieces_all, first, done)
    return state


  def step(self, state, player, action):
    # player_list and remaining_pieces_all required in order to check if game has ended.
    # player = 1 ~ 4
    # action = (piece_id, i, j, rotation, flip)
    # this assumes a valid action
    next_board = state.board.copy()
    remaining_pieces_all = deepcopy(state.remaining_pieces_all)
    first = deepcopy(state.first)
    done = deepcopy(state.done)

    can_place = self.place_possible(state.board, player, action)
    if not can_place:
      raise ValueError(f"You can't place here.\nplayer: {player}\naction: {action}\nboard:\n{state.board}")

    idx, i, j, rot, flip = action
    
    # later change back
    block, corners, neighbors, diagonals = self.__adjust(self.pieces[idx], rot, flip)
    # block, corners, neighbors, diagonals = self.pieces[idx][(rot, flip)]

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

    remaining_pieces_all[player].remove(action[0])
    next_player = self.next_player(player)
    next_state = State(next_board, remaining_pieces_all, first, done)

    # next_actions = self.possible_actions(next_state, player)
    # if len(next_actions) == 0:
    #   next_state.done[player] = True
    game_done = self.check_game_finished(next_state)
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
    actions = []
    # for all blocks
    for idx in state.remaining_pieces_all[player]:
      piece = self.pieces[idx]
      # for all rotation/flip
      for (r, f), data in piece.items():
        block, corners, neighbors, diagonals = data#data['block'], data['corners'], data['neighbors'], data['diagonals']
        width, height = len(block[0]), len(block)
        # for all diagonals (on the board)
        for diag_pos in np.argwhere(state.board[player]):
          # for all corners (on piece)
          for offset in np.argwhere(corners):
            pos = diag_pos - offset
            # if coord goes off the board, ignore and continue
            if np.any(pos < 0) or np.any(pos+[height, width] > self.size):
              continue
            else:
              i, j = pos
              action = (idx, i, j, r, f)
              if self.place_possible(state.board, player, action):
                actions.append(action)
    return actions

  def place_possible(self, board, player, action):
    idx, i, j, rot, flip = action

    # later change back
    piece = self.pieces[idx]
    block, corners, neighbors, diagonals = self.__adjust(self.pieces[idx], rot, flip)
    # block, corners, neighbors, diagonals = piece[(rot, flip)]
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

  def check_player_finished(self, state, player):
    for idx in state.remaining_pieces_all[player]:
      piece = self.pieces[idx]
      for (r, f), data in piece.items():
        block, corners, neighbors, diagonals = data
        width, height = len(block[0]), len(block)
        for diag_pos in np.argwhere(state.board[player]):
          for offset in np.argwhere(corners):
            pos = diag_pos - offset
            if np.any(pos < 0) or np.any(pos+[height, width] > self.size):
              continue
            else:
              i, j = pos
              action = (idx, i, j, r, f)
              if self.place_possible(state.board, player, action):
                return False
    return True

  def check_game_finished(self, state):
    for player in self.player_list:
      if not state.done[player]:
        if not self.check_player_finished(state, player):
          return False
        state.done[player] = True
    return True
    # 꼭지점이 있어도 게임이 끝나는 경우가 많아서 이 방식을 사용하려면 플레이어가 액션이 더이상 없을때 state에 diagonal을 전부 없애주는 식으로 해야함.
    # 근데 우선은 그냥 이 함수 내에서 모든 플레이어에 대해 끝났는지 확인하고 있음.
    # return: WIN = 1, LOSE = -1, UNDETERMINED = 0
    # done IFF nobody has a diagonal.
    # i.e., even if the current player can't play anymore, if others can, the game is not done.
    # ----------
    # if np.sum(state[1:1+self.num_players]) == 0:
    #   scores = []
    #   for idx in self.player_list:
    #     score = np.sum(state[0] == idx)
    #     scores.append(score)
    #   rank = np.argsort(scores).argsort()
    #   # [0] -- dummy to make indexing easier
    #   reward = [0]
    #   for i in rank:
    #     if i == 0:
    #       reward.append(-1)
    #     elif i == 1:
    #       reward.append(1)
    #   return True, reward
    # return False, [0,0,0]

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

  def next_player(self, player):
    if player == self.player_list[-1]:
      return self.player_list[0]
    idx = self.player_list.index(player)
    return self.player_list[idx + 1]


# if __name__ == '__main__':
#   env = Blokus(8, [1,2])
#   total = 0
#   for piece in env.pieces.values():
#     total += len(piece['rotflip'])
#   print(total)