import sys, os
from collections import namedtuple

Player = namedtuple('Player', ['mode', 'idx', 'name'])

class Board:
    def __init__(self, state):
        '''This class is purely for visualization'''
        self.state = state.board[:21].sum(axis=0) + 2*state.board[21:42].sum(axis=0)
        self.template = {0: '.', 1: 'O', 2: 'X'}
        self.size = len(self.state)
        self.top = ' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'[:self.size * 2] + '\n'
        row = ' '.join(['{}'] * self.size)
        rows = [str(i).rjust(2) + ' ' + row for i in range(self.size)]
        self.total = '  ' + self.top + '\n'.join(rows)

    def __repr__(self):
        l = [self.template[num] for num in self.state.reshape(1,-1)[0]]
        return self.total.format(*l)

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
        p1_name = input("Enter the name for Player 1: ").strip()
        p2_name = input("Enter the name for Player 2: ").strip()
        blue = Player('human', 0, f'(Player 1) {p1_name}')
        yellow = Player('human', 1, f'(Player 2) {p2_name}')

    elif option == 2:
        input_ = input('Play first? [y/n] ').strip().lower()
        if input_ == 'q':
            sys.exit()
        if input_ in ['y', 'yes']:
            blue = Player('human', 0, 'Player')
            yellow = Player('mcts', 1, 'Computer')
        elif input_ in ['n', 'no']:
            blue = Player('mcts', 0, 'Computer')
            yellow = Player('human', 1, 'Player')
        else:
            raise ValueError('Invalid input.')

    elif option == 3:
        blue = Player('mcts', 0, 'Computer 1')
        yellow = Player('mcts', 1, 'Computer 2')
    return blue, yellow

def action_helper(actions, threshold):
    if len(actions) < threshold:
        print('actions:')
        for action in actions:
            print(action)

def turn_helper(turn, state):
    os.system('clear')
    print(f'Turn {turn}:')
    board = Board(state)
    print(board, end='\n\n')

def action_wrapper(color, state, actions, env, mcts):
    remaining_pieces = env.get_remaining_pieces(state, color.idx)
    if color.mode == 'human':
        while True:
            action = ask_action(actions)
            if action[0] in remaining_pieces and env.place_possible(state.board, color.idx, action):
                break
            else:
                print('That\'s not possible. Choose another place')
    elif color.mode == 'mcts':
        action = mcts.get_action(state)
    print('Action chosen:', action)
    return action
