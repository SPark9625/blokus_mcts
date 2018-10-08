import datetime
import multiprocessing
import random
import sys

import numpy as np

from mcts import UCT
from model import Blokus
from util import *

# action = (idx, i, j, r, f)
# 219, 411, 443, 475


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    BOARD_SIZE = 13

    TIME_BUDGET = 30
    ITER_BUDGET = 400

    NUM_WORKERS = 2

    blue, yellow = determine_roles()
    player_list = [blue.idx, yellow.idx]

    env = Blokus(board_size=BOARD_SIZE, player_list=player_list)
    mcts = UCT(env, player_list, num_workers=NUM_WORKERS, time_budget=TIME_BUDGET, iter_budget=ITER_BUDGET, exploration=1.4)

    turn = 0
    state = env.reset()

    get_pieces_slice = {blue.idx: slice(21), yellow.idx: slice(21,42)}


    while True:
        turn_helper(turn, state)
        actions = env.possible_actions(state)
        player = state.board[env.TURN, 0, 0]

        # get action 
        color = blue if player == blue.idx else yellow
        print(f'{color.name}\'s turn. {len(actions)} actions left')
        remaining_pieces = env.get_remaining_pieces(state, player)
        print('Remaining pieces:\n', remaining_pieces)

        # Helper functionality -- prints out actions if the number of possible actions is below a certain threshold
        action_helper(actions, threshold=10)

        # Returns an action either by asking (if mode is 'human'), or by running the mcts agent.
        action = action_wrapper(color, state, actions, env, mcts)

        # -------------------------- #
        #         env.step()         #
        # -------------------------- #
        state, reward, done, _ = env.step(state, action, actions)

        # -------------------------- #
        #        bookkeeping         #
        # -------------------------- #
        turn += 1
        if done:
            turn_helper(turn, state)
            print('Game finished. Rewards:', reward)
            break
