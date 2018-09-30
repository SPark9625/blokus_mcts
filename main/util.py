import sys, os
from collections import namedtuple

Player = namedtuple('Player', ['mode', 'idx', 'name'])

class Board:
  def __init__(self, state):
    '''This class is purely for visualization'''
    self.state = state
    self.template = {0: '.', 1: 'O', 2: 'X'}
    self.size = len(state)
    row = '[' + ' '.join(['{}'] * self.size) + ']'
    self.total = '\n'.join([row] * self.size)

  def __repr__(self):
    l = [self.template[num] for num in self.state.reshape(1,-1)[0]]
    return self.total.format(*l)

def switch_player(player, player_list):
  if player == player_list[-1]:
    return player_list[0]
  return player + 1

def ask_action(actions):
  while True:
    input_ = input('Input an action in the format (piece_id, i, j, r, f): ').strip().lower()
    if input_ == 'q':
      sys.exit()
    elif input_.startswith('!'):
      try:
        eval(input_[1:])
        continue
      except:
        continue
    try:
      action = eval(input_)
      assert isinstance(action, tuple) and len(action) == 5
      break
    except:
      print('That\'s not a valid action input. Try again.')
  return action

def determine_roles():
  os.system('clear')
  input_ = input('Choose an option:\n1. human vs human\n2. MCTS vs human\n3. MCTS vs MCTS\n\nYour choice: ').strip().lower()
  if input_ == 'q':
    sys.exit()
  option = int(input_)
  if option == 1:
    blue = Player('human', 1, 'Player 1')
    yellow = Player('human', 2, 'Player 2')
  elif option == 2:
    input_ = input('Play first? [y/n] ').strip().lower()
    if input_ == 'q':
      sys.exit()
    if input_ in ['y', 'yes']:
      blue = Player('human', 1, 'Player')
      yellow = Player('mcts', 2, 'Computer')
    elif input_ in ['n', 'no']:
      blue = Player('mcts', 1, 'Computer')
      yellow = Player('human', 2, 'Player')
    else:
      raise ValueError('Invalid input.')
  elif option == 3:
    blue = Player('mcts', 1, 'Computer 1')
    yellow = Player('mcts', 2, 'Computer 2')
  return blue, yellow

def action_helper(actions, threshold):
  if len(actions) < threshold:
    print('actions:')
    for action in actions:
      print(action)

def turn_helper(turn, state):
  os.system('clear')
  print(f'Turn {turn}:')
  board = Board(state.board[0])
  print(board, end='\n\n')