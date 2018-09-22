import random
from model import Blokus
from mcts import UCT
import datetime
from util import *
# action = (idx, i, j, r, f)





if __name__ == '__main__':
  BOARD_SIZE = 5
  BLUE, YELLOW = 1, 2
  num_players = 2

  player_list = [BLUE, YELLOW]
  human, computer = determine_roles(player_list)

  env = Blokus(size=BOARD_SIZE, player_list=player_list)
  mcts = UCT(env, computer, player_list, budget=60, iter_budget=1600, exploration=1.4)
  # mcts.initialize()
  
  


  turn = 0
  state = env.reset()
  cur_player = player_list[0]
  print(f'Initial state:')
  print(state.board)



  while True:
    actions = env.possible_actions(state, cur_player)

    # This player is finished.
    # This does not mean that the game has finished though, so just switch player and continue.
    if len(actions) == 0:
      print('Skipping')
      cur_player = switch_player(cur_player, player_list)
      continue

    # get action from player
    elif cur_player == human:
      print(f'Player\'s turn. You have {len(actions)} actions left')

      # Helper functionality
      action_helper(actions, threshold=10)

      while True:
        action = ask_action()
        if env.place_possible(state.board, cur_player, action):
          break
        else:
          print('That\'s not possible. Choose another place')


    # get action from MCTS
    else:
      print(f'Computer\'s turn. Computer has {len(actions)} actions left')
      action_helper(actions, threshold=10)

      while True:
        action = ask_action()
        if env.place_possible(state.board, cur_player, action):
          break
        else:
          print('That\'s not possible. Choose another place')
      # print('Start:',datetime.datetime.now())
      # action = mcts.get_action(state, remaining_pieces_all)

      print('Computer action:', action)


    state, reward, done, _ = env.step(state, cur_player, action)




    # bookkeeping
    turn += 1
    cur_player = switch_player(cur_player, player_list)


    print(f'Turn {turn}:')
    print(state.board, end='\n\n')

    if done:
      print('Game finished. Rewards:', reward)
      break




