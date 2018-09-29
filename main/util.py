import sys

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

def determine_roles(player_list):
  input_ = input('Play first? [y/n] ').strip().lower()
  if input_ == 'q':
    sys.exit()
  human = player_list[0] if input_ in ['y', 'yes'] else player_list[1]
  computer = player_list[1] if human == player_list[0] else player_list[0]
  return human, computer

def action_helper(actions, threshold):
  if len(actions) < threshold:
    print('actions:')
    for action in actions:
      print(action)

def turn_helper(turn, state):
  print(f'Turn {turn}:')
  board = Board(state.board[0])
  print(board, end='\n\n')