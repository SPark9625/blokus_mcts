import sys

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