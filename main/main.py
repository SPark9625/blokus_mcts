"""
This file lets you play:
(No training involved)
    1. Human vs Human
    2. Human vs Agent
    3. Agent vs Agent
"""

import datetime
import multiprocessing
import random
import sys

import numpy as np

# from mcts_naive import UCT
# from mcts_agz import MCTS
from model import Blokus
from config import config
import util

if __name__ == '__main__':
    blue, yellow = util.determine_roles()

    env = Blokus()
    # mcts = UCT(env, exploration=1.4)
    mcts = None
    if blue.mode == 'mcts' or yellow.mode == 'mcts':
        print('Creating agent...')
        mcts = MCTS(env, tau=1).eval()


    turn = 0
    state = env.reset()
    while True:
        player = state.board[-1, 0, 0]
        actions = state.meta.actions[player]
        color = blue if player == blue.idx else yellow
        remaining_pieces = env.get_remaining_pieces(state, player)

        util.turn_helper(turn, state, color, remaining_pieces)
        

        # Print actions if len(actions) < threshold
        util.action_helper(actions, threshold=10)

        # Get action either from human or from the agent
        action = util.action_wrapper(color, state, env, mcts)

        state, reward, done, _ = env.step(state, action, actions)

        turn += 1
        if done:
            util.turn_helper(turn, state, done=True)
            print('Game finished. Rewards:', reward)
            break
