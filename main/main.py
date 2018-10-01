import random, sys
from model import Blokus
from mcts import UCT
import datetime
from util import *
# action = (idx, i, j, r, f)
# 219, 411, 443, 475


if __name__ == '__main__':
    BOARD_SIZE = 13

    TIME_BUDGET = 10
    ITER_BUDGET = 1600

    blue, yellow = determine_roles()
    player_list = [blue.idx, yellow.idx]

    env = Blokus(size=BOARD_SIZE, player_list=player_list)
    mcts = UCT(env, player_list, num_workers=2, time_budget=TIME_BUDGET, iter_budget=ITER_BUDGET, exploration=1.4)

    turn = 0
    state = env.reset()


    while True:
        turn_helper(turn, state)
        actions = env.possible_actions(state, state.player)

        # ---------------------------- #
        #                              #
        #     get action from blue     #
        #                              #
        # ---------------------------- #
        if state.player == blue.idx:
            print(f'{blue.name}\'s turn. {len(actions)} actions left')
            print('Remaining pieces:\n', state.remaining_pieces_all[state.player])

            # Helper functionality -- prints out actions if the number of possible actions is below a certain threshold
            action_helper(actions, threshold=10)

            # Returns an action either by asking (if mode is 'human'), or by running the mcts agent.
            action = action_wrapper(blue, state, actions, env, mcts)

        # ---------------------------- #
        #                              #
        #    get action from yellow    #
        #                              #
        # ---------------------------- #
        elif state.player == yellow.idx:
            print(f'{yellow.name}\'s turn. {len(actions)} actions left')
            print('Remaining pieces:\n', state.remaining_pieces_all[state.player])

            # Helper functionality
            action_helper(actions, threshold=10)
            action = action_wrapper(yellow, state, actions, env, mcts)

        else:
            raise ValueError('Something went (terribly) wrong.')


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




