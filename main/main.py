import random, sys
from model import Blokus
from mcts import UCT
import datetime
from util import *
# action = (idx, i, j, r, f)


computer_mode = 'mcts'
BOARD_SIZE = 5

if __name__ == '__main__':
  TIME_BUDGET = 60
  ITER_BUDGET = 1600

  BLUE, YELLOW = 1, 2
  num_players = 2

  player_list = [BLUE, YELLOW]
  human, computer = determine_roles(player_list)

  env = Blokus(size=BOARD_SIZE, player_list=player_list)
  mcts = UCT(env, computer, player_list, time_budget=TIME_BUDGET, iter_budget=ITER_BUDGET, exploration=1.4)
  # mcts.initialize()



  turn = 0
  state = env.reset()


  while True:
    turn_helper(turn, state)
    actions = env.possible_actions(state, state.player)

    # -------------------------- #
    #                            #
    #   get action from player   #
    #                            #
    # -------------------------- #
    if state.player == human:
      print(f'Player\'s turn. You have {len(actions)} actions left')
      print('Your pieces:', state.remaining_pieces_all[state.player])

      # Helper functionality
      action_helper(actions, threshold=10)

      while True:
        action = ask_action(actions)
        if action[0] in state.remaining_pieces_all[state.player] and env.place_possible(state.board, state.player, action):
          break
        else:
          print('That\'s not possible. Choose another place')

    # -------------------------- #
    #                            #
    #    get action from MCTS    #
    #                            #
    # -------------------------- #
    else:
      print(f'Computer\'s turn. Computer has {len(actions)} actions left')
      print('Computer\'s pieces:', state.remaining_pieces_all[state.player])
      action_helper(actions, threshold=10)

      if computer_mode == 'human':
        while True:
          action = ask_action(actions)
          if action[0] in state.remaining_pieces_all[state.player] and env.place_possible(state.board, state.player, action):
            break
          else:
            print('That\'s not possible. Choose another place')
      else:
        action = mcts.get_action(state)

      print('Computer action:', action)


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




