import random, sys
from model import Blokus
from mcts import UCT
import datetime
from util import *
# action = (idx, i, j, r, f)
# 219, 411, 443, 475
BOARD_SIZE = 13

if __name__ == '__main__':
  TIME_BUDGET = 60 * 15
  ITER_BUDGET = 1600
  
  blue, yellow = determine_roles()
  player_1, player_2 = blue.idx, yellow.idx
  player_list = [player_1, player_2]

  env = Blokus(size=BOARD_SIZE, player_list=player_list)
  mcts = UCT(env, player_list, num_workers=2, time_budget=TIME_BUDGET, iter_budget=ITER_BUDGET, exploration=1.4)
  # mcts.initialize()



  turn = 0
  state = env.reset()


  while True:
    turn_helper(turn, state)
    actions = env.possible_actions(state, state.player)

    # ---------------------------- #
    #                              #
    #   get action from player_1   #
    #                              #
    # ---------------------------- #
    if state.player == blue.idx:
      print(f'{blue.name}\'s turn. {len(actions)} actions left')
      print('Remaining pieces:\n', state.remaining_pieces_all[state.player])

      # Helper functionality
      action_helper(actions, threshold=10)

      if blue.mode == 'human':
        while True:
          action = ask_action(actions)
          if action[0] in state.remaining_pieces_all[state.player] and env.place_possible(state.board, state.player, action):
            break
          else:
            print('That\'s not possible. Choose another place')
      elif blue.mode == 'mcts':
        action = mcts.get_action(state)
      print('Action chosen:', action)

    # ---------------------------- #
    #                              #
    #   get action from player_2   #
    #                              #
    # ---------------------------- #
    elif state.player == yellow.idx:
      print(f'{yellow.name}\'s turn. {len(actions)} actions left')
      print('Remaining pieces:\n', state.remaining_pieces_all[state.player])

      # Helper functionality
      action_helper(actions, threshold=10)

      if yellow.mode == 'human':
        while True:
          action = ask_action(actions)
          if action[0] in state.remaining_pieces_all[state.player] and env.place_possible(state.board, state.player, action):
            break
          else:
            print('That\'s not possible. Choose another place')
      elif yellow.mode == 'mcts':
        action = mcts.get_action(state)
      print('Action chosen:', action)

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




