from collections import namedtuple
from copy import deepcopy

import numpy as np

from config import config

State = namedtuple('State', ['board', 'meta'])
Meta = namedtuple('Meta', ['actions', 'new_diagonals'])

class Blokus:
    def __init__(self):
        self.pieces = config.pieces

        self.SIZE           = config.board_size
        self.player_list    = config.player_list
        self.N_PLAYERS      = config.num_players
        self.N_PIECES       = config.num_pieces
        self.N_STATE_LAYER  = config.num_state_layers
        self.N_ACTION_LAYER = config.num_action_layers
        self.BOARD_SHAPE    = config.board_shape
        self.ACTION_SHAPE   = config.action_shape

        self.layer2irf = config.layer2irf
        self.irf2layer = config.irf2layer
        self.idx2slice = config.idx2slice

        self.DIAGONAL = self.N_PLAYERS * self.N_PIECES
        self.NEIGHBOR = self.DIAGONAL + self.N_PLAYERS
        self.FIRST    = self.NEIGHBOR + self.N_PLAYERS
        self.DONE     = self.FIRST + self.N_PLAYERS
        self.TURN     = -1


    def reset(self):
        '''Assuming two players:
            - board[  :21]: Player 1's pieces. np.zeros() if piece hasn't been used
            - board[21:42]: Player 2's pieces.

            - board[42]: Player 1's diagonal positions.
            - board[43]: Player 2's diagonal positions.
            
            - board[44]: Player 1's neighboring positions
            - board[45]: Player 2's neighboring positions.
            
            - board[46]: if Player 1 is placing for the first time
            - board[47]: if Player 2 is placing for the first time
            
            - board[48]: if Player 1 is finished
            - board[49]: if Player 2 is finished
            
            - board[50]: Who's turn.

        isinstance(player, int) == True
        isinstance(player_list, list) == True
        '''
        board = np.zeros(self.BOARD_SHAPE, dtype=np.int8)

        first_pos = np.array([[[0,0]],[[self.SIZE-1,self.SIZE-1]]])
        for p in self.player_list:
            # `diagonal` layers
            board[self.DIAGONAL + p][tuple(first_pos[p,0])] = 1
            # Set `first` to True
            board[self.FIRST + p] = 1

        # To speed up calculation in `possible_actions` and `_check_player_finished`
        new_diagonals_all = [first_pos[p] for p in self.player_list]
        actions_all       = [[] for p in self.player_list]
        meta = Meta(actions_all, new_diagonals_all)

        state = State(board, meta)
        actions = self.possible_actions(state, self.player_list[0])
        state.meta.actions[self.player_list[0]] = actions
        return state


    def step(self, state, action, actions):
        """
        Arguments:
            state {np.ndarray} -- self.NUM_STATE_LAYERS * self.SIZE * self.SIZE
            action {tuple} -- rank 5 or 3
            actions {np.ndarray} -- list of actions
        
        Raises:
            ValueError -- if action is invalid
        
        Returns:
            next_state, reward, done, info -- follows OpenAI's gym API.
        """

        # action = (piece_id, rotation, flip, i, j) *OR*
        # action = (layer, i, j)
        # this assumes a valid action
        next_board = state.board.copy()
        player = next_board[self.TURN, 0, 0]

        can_place = self.place_possible(state.board, player, action)
        if not can_place:
            raise ValueError(f"You can't place here.\nplayer: {player}\naction: {action}\nboard:\n{state.board}")

        if len(action) == 5:
            idx, r, f, i, j = action
        elif len(action) == 3:
            layer, i, j = action
            idx, r, f = self.layer2irf[layer]
        piece = self.pieces[idx]

        if (r, f) in piece.keys():
            block, corners, neighbors, diagonals = deepcopy(piece[(r, f)])
        else:
            block, corners, neighbors, diagonals = self._adjust(piece, r, f)

        width, height = len(block[0]), len(block)
        # update main board: assumes a valid action, thus just add.
        next_board[self.N_PIECES*player + idx, i:i+height, j:j+width] += block

        first = next_board[self.FIRST + player, 0, 0]
        if first:
            next_board[self.DIAGONAL + player,[0,0,self.SIZE-1,self.SIZE-1],[0,self.SIZE-1,0,self.SIZE-1]] = 0
            next_board[self.FIRST + player] = int(False)

        # calculate outer slices
        diagonals, neighbors, x_outer_slice, y_outer_slice = self.fit_to_board(
            i, j, height, width, diagonals, neighbors)

        # ----------------------- #
        #   1. Update neighbors   #
        #   2. Update diagonals   #
        #   3. Add diagonals      #
        # ----------------------- #

        # ------------------- #
        # 1. Update neighbors #
        # ------------------- #
        existing_neighbors = next_board[self.NEIGHBOR + player, y_outer_slice, x_outer_slice]
        neighbors[np.logical_and(existing_neighbors, neighbors)] = 0
        existing_neighbors += neighbors

        # ------------------- #
        # 2. Update diagonals #
        # ------------------- #
        # This has to be perfectly accurate at all times.
        # First, remove any existing diagonals where a piece is about to be placed.
        focus = next_board[self.DIAGONAL:self.DIAGONAL + self.N_PLAYERS, i:i+height, j:j+width]
        focus[np.logical_and(focus, block)] = 0

        outer_diagonals = next_board[self.DIAGONAL + player, y_outer_slice, x_outer_slice]
        outer_diagonals[np.logical_and(outer_diagonals, neighbors)] = 0    

        # If places where the new piece is about to declare diagonal are already placed, they can't be.
        main_board = next_board[:self.DIAGONAL, y_outer_slice, x_outer_slice]
        diagonals[np.logical_and(main_board, diagonals).sum(axis=0, dtype=bool)] = 0

        # same with places adjacent to the player's own color
        neighbor_board = next_board[self.NEIGHBOR + player, y_outer_slice, x_outer_slice]
        diagonals[np.logical_and(neighbor_board, diagonals)] = 0

        # I take an additional step here to subtract places that are already declared diagonal.
        # This is so that I don't add places to `new_diagonals` when they aren't actually new.
        diagonals[np.logical_and(outer_diagonals, diagonals)] = 0

        # ------------------ #
        #  3. Add diagonals  #
        # ------------------ #
        outer_diagonals += diagonals
        offset = [y_outer_slice.start, x_outer_slice.start]
        new_diagonals = np.argwhere(diagonals) + offset  # (N, 2) np.ndarray

        # For metadata
        actions_all = deepcopy(state.meta.actions)
        new_diagonals_all = deepcopy(state.meta.new_diagonals)

        actions = [a for a in actions if a[0] != idx]
        actions_all[player] = actions
        new_diagonals_all[player] = new_diagonals

        meta = Meta(actions_all, new_diagonals_all)
        next_state = State(next_board, meta)

        game_done = self._check_game_finished(next_state)


        # if current player is not done, this just returns the next player that
        # wasn't finished at the beginning of the current player's turn
        if not game_done:
            next_board[self.TURN] = self.next_player(next_board)

        

        reward = [0 for p in self.player_list]  # game hasn't finished, or is a draw
        if game_done:
            scores = [next_board[p*self.N_PIECES:(p+1)*self.N_PIECES].sum() for p in self.player_list]
            if scores[0] > scores[1]:
                reward = [1, -1]
            elif scores[0] < scores[1]:
                reward = [-1, 1]
        return next_state, reward, game_done, None

    def possible_actions(self, state, player):
        """
        Calculates all possible actions for a player.

        Arguments:
            state {np.ndarray} -- self.NUM_STATE_LAYERS * self.SIZE * self.SIZE
        
        Returns:
            all possible actions
        """

        # disabled actions can be quickly removed and only additional diagonals are checked thoroughly
        prev_actions = state.meta.actions[player]  # list of 3 dim tuples

        i_s, j_s = state.meta.new_diagonals[player].T
        remaining_pieces = self.get_remaining_pieces(state, player)

        dead_actions = set()
        for action in prev_actions:
            idx = self.layer2irf[action[0]][0]
            if idx not in remaining_pieces or not self.place_possible(state.board, player, action):
                dead_actions.add(action)
        prev_actions = set(prev_actions) - dead_actions

        alive = state.board[self.DIAGONAL + player, i_s, j_s].astype(bool)
        new_diagonals = state.meta.new_diagonals[player][alive]

        new_actions = set()
        for diag_pos in new_diagonals:
            for idx in remaining_pieces:
                piece = self.pieces[idx]
                for (r, f), data in piece.items():
                        block, corners = data[:2]
                        width, height = len(block[0]), len(block)
                        for offset in np.argwhere(corners):
                            pos = diag_pos - offset
                            # if coord goes off the board, ignore and continue
                            if np.any(pos < 0) or np.any(pos+[height, width] > self.SIZE):
                                continue
                            else:
                                i, j = pos
                                layer = self.irf2layer[(idx, r, f)]
                                action = (layer, i, j)
                                if self.place_possible(state.board, player, action):
                                    new_actions.add(action)
        actions = list(prev_actions | new_actions)
        return actions

    def get_remaining_pieces(self, state, player):
        player_pieces_slice = slice(
            player*self.N_PIECES, (player+1)*self.N_PIECES)
        used = state.board[player_pieces_slice].any(axis=(1, 2))
        remaining_pieces = np.argwhere(~used).flatten()
        return remaining_pieces

    def place_possible(self, board, player, action):
        if len(action) == 5:
            idx, r, f, i, j = action
        elif len(action) == 3:
            layer, i, j = action
            idx, r, f = self.layer2irf[layer]

        piece = self.pieces[idx]
        if (r, f) in piece.keys():
            block = piece[(r, f)][0]
        else:
            block = self._adjust(piece, r, f)[0]
        width, height = len(block[0]), len(block)

        # check overlap
        if np.logical_and(board[:self.DIAGONAL, i:i+height, j:j+width], block).any():
            return False

        # check if there are any common flat edge
        if np.logical_and(board[self.NEIGHBOR + player, i:i+height, j:j+width], block).any():
            return False

        # make sure the corners touch
        if np.logical_and(board[self.DIAGONAL + player, i:i+height, j:j+width], block).any():
            return True

        return False

    def _check_player_finished(self, state, player):
        board, meta = state
        prev_actions = meta.actions[player]
        i_s,j_s = meta.new_diagonals[player].T

        remaining_pieces = self.get_remaining_pieces(state, player)
        for action in prev_actions:
            idx = self.layer2irf[action[0]][0]
            if idx in remaining_pieces and self.place_possible(board, player, action):
                return False


        alive = board[self.DIAGONAL + player, i_s, j_s].astype(bool)
        new_diagonals = meta.new_diagonals[player][alive]
        for idx in remaining_pieces:
            piece = self.pieces[idx]
            for (r, f), data in piece.items():
                block, corners = data[:2]
                width, height = len(block[0]), len(block)
                for diag_pos in new_diagonals:
                    for offset in np.argwhere(corners):
                        pos = diag_pos - offset
                        # if coord goes off the board, ignore and continue
                        if np.any(pos < 0) or np.any(pos+[height, width] > self.SIZE):
                            continue
                        else:
                            i, j = pos
                            action = (idx, r, f, i, j)
                            if self.place_possible(board, player, action):
                                return False
        return True

    def _check_game_finished(self, state):
        # if A places and makes B,C,... all finished without finishing herself, the situation becomes:
        # A = not done, B = C = ... = done
        # But since this function returns False if the current player is not done, what actually happens is:
        # A = B = C = ... = not done
        board = state[0]
        finished = True

        cur = board[self.TURN, 0, 0]
        next_flag = False  # When checking next player, calculate all possible moves
        for p in self.player_list:
            player = (cur + p) % self.N_PLAYERS  # start from the currect player
            if not board[self.DONE + player, 0, 0]:
                # Calculate if player has any actions left
                if next_flag:
                    next_flag = False
                    actions = self.possible_actions(state, player)
                    state.meta.actions[player] = actions
                    if len(actions) > 0:
                        finished = False
                        break
                    else:
                        board[self.DONE + player] = 1  # set done to True
                        continue
                else:
                    if self._check_player_finished(state, player):
                        board[self.DONE + player] = 1  # set done to True
                    else:
                        finished = False
                        if player != cur:
                            break
                    if player == cur:
                        next_flag = True
        return finished

    @staticmethod
    def _adjust(piece, r, f):
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


    def fit_to_board(self, i, j, height, width, diagonals, neighbors):
        x_left, y_top = j-1,i-1
        x_right, y_bottom = j+width+1, i+height+1
        if x_left < 0:
            diff = -x_left
            x_left = 0
            neighbors = neighbors[:, diff:]
            diagonals = diagonals[:, diff:]

        if x_right > self.SIZE:
            diff = x_right - self.SIZE
            x_right = self.SIZE
            neighbors = neighbors[:, :-diff]
            diagonals = diagonals[:, :-diff]
          
        if y_top < 0:
            diff = -y_top
            y_top = 0
            neighbors = neighbors[diff:]
            diagonals = diagonals[diff:]
        if y_bottom > self.SIZE:
            diff = y_bottom - self.SIZE
            y_bottom = self.SIZE
            neighbors = neighbors[:-diff]
            diagonals = diagonals[:-diff]
        x_slice = slice(x_left, x_right)
        y_slice = slice(y_top, y_bottom)
        return diagonals, neighbors, x_slice, y_slice

    def next_player(self, board):
        """
        This assumes that players are ordered integers.
        If no one is alive, returns None
        """
        nex = board[self.TURN, 0, 0] + 1
        for p in self.player_list:
            player = (nex + p) % self.N_PLAYERS  # start from the next player
            if board[self.DONE + player, 0, 0] == 0:
                return player

if __name__ == '__main__':
    env = Blokus()
    # while True:
    #     eval(input())
