import time
from math import sqrt, log
import random
from copy import deepcopy
import pickle

BLUE, YELLOW = 1, 2


class Node:
  def __init__(self, state, player, parent, action_in, terminal=False, result=None, depth=None):
    self.depth = depth
    self.state = state
    self.parent = parent
    self.action_in = action_in

    self.children = []

    self.value = 0
    self.n_visit = 0

    self.terminal = terminal
    self.result = result

    self.player = player

  @property
  def mean(self):
    return self.value/self.n_visit

  def print(self):
    print(f'Node {self.depth}:\n\taction_in: {self.action_in},\n\tnum_children: {len(self.children)},\
\n\tterminal: {self.terminal},\n\tresult: {self.result},\n\tplayer: {self.player}\n\tstate:\n{self.state[0]}')

  def __repr__(self):
    return f'{self.action_in}: {self.value}/{self.n_visit} (={str(self.mean)[:5]})'

    


class UCT:
  def __init__(self, env, player, player_list, budget, iter_budget=1600, exploration=1.4):
    """env has to have 2 APIs: env.step() and env.possible_actions()
      env.step() takes (state, action, player) as input and has to return: (next_state, reward, done, info)
      env.possible_actions() takes `state`, `player_id`, `piece_idx` as input and has to return a `list` of actions.
    """
    self.env = env
    self.player = player
    self.player_list = player_list
    self.opponent = BLUE if player == YELLOW else YELLOW

    self.budget = budget  # seconds
    self.iter_budget = iter_budget
    
    self.exploration = exploration
    self.nodes = {}

  def initialize(self):
    size = self.env.size
    try:
      fname = f'{size}x{size}.pkl'
      with open(fname, 'rb') as f:
        self.nodes = pickle.load(f)
        print(f'Reading tree data from {fname}.')
    except FileNotFoundError:
      s0 = self.env.reset()
      root = Node(s0, self.player_list[0], parent=None, action_in=None, depth=0, cur_root=True)
      self.nodes[s0.tobytes()] = root
      print(f'Initialized a new tree of size {size}x{size}')

  def get_node(self, s0, player_id):
    try:
      node = self.nodes[s0.tobytes()]
      parent = node.parent
      while parent:
        self.nodes.pop(parent.state.tobytes())
        parent = parent.parent
      node.parent = None
      return node

    except KeyError:
      # node = Node(s0, player=player_id, parent=None, action_in=None, depth=0)
      # self.nodes[s0.tobytes()] = node
      # return node
      raise KeyError('Did you forget to call <UCT>.initialize()')

  def get_action(self, state, piece_list):
    cur_root = self.get_node(state, self.player)
    t0 = time.time()
    iteration = 0
    while time.time() - t0 < self.budget and iteration < self.iter_budget:
      piece_list_copy = deepcopy(piece_list)
      # print('entering tree_policy')
      leaf_node = self.tree_policy(cur_root, piece_list_copy)
      # print()
      # leaf_node.print()
      # print()
      # print('finished tree_policy')
      
      if not leaf_node.terminal:
        # print('entering default_policy')
        result = self.default_policy(leaf_node, piece_list_copy)
        # print('finished default_policy')
      else:
        # print('getting leaf_node.result')
        result = leaf_node.result
        # print('leaf_node.result is:', result)
      # print('backing up')
      self.backup(leaf_node, result)
      # print('backed up')
      iteration += 1
      print('iteration:',iteration)
      # print('------------------')

    print()
    cur_root.children.sort(key=lambda x:x.mean, reverse=True)
    for child in cur_root.children:
      print(child)
    print(f'Finished {iteration} iterations.')
    best_node = self.best_child(cur_root)
    action = best_node.action_in
    return action

  def tree_policy(self, node, piece_list):
    while not node.terminal:
      actions = self.env.possible_actions(node.state, node.player, piece_list[node.player])

      if len(actions) == 0:
        # done, result = self.env.check_game_finished(node.state)
        # if done:
        #   node.terminal = True
        #   node.result = result
        #   break
        # else:
        node.player = self.switch_player(node.player)
        continue

      elif 0 < len(node.children) < len(actions):
        # not fully expanded
        return self.expand(node, actions, piece_list)

      else:
        # fully expanded
        player_in = node.player
        node = self.best_child(node, self.exploration)
        piece_list[player_in].remove(node.action_in[0])
        

    return node

  def default_policy(self, node, piece_list):
    terminal = node.terminal
    assert not terminal
    state = node.state
    player = node.player
    
    dummy = 0
    while not terminal:
      actions = self.env.possible_actions(state, player, piece_list[player])
      if len(actions) == 0:
        # print(f'player {player} has no more actions')
        player = self.switch_player(player)
        continue
      action = random.choice(actions)
      first = True if len(piece_list[player]) == len(self.env.pieces) else False
      state, result, terminal, info = self.env.step(state, action, player, self.player_list, piece_list, first=first)

      piece_list[player].remove(action[0])
      player = self.switch_player(player)
    return result

  def expand(self, node, actions, piece_list):
    for child in node.children:
      if child.action_in in actions:
        actions.remove(child.action_in)

    action = random.choice(actions)
    first = True if len(piece_list[node.player]) == len(self.env.pieces) else False
    next_state, reward, done, info = self.env.step(node.state, action, node.player, self.player_list, piece_list, first=first)
    piece_list[node.player].remove(action[0])

    player = self.switch_player(node.player)
    kwargs = dict(
        state=next_state,
        player=player,
        parent=node,
        action_in=action,
        depth=node.depth + 1,
        terminal=False,
        result=None,
      )
    if done:
      kwargs['terminal'] = True
      kwargs['result'] = reward
    
    next_node = Node(**kwargs)
    node.children.append(next_node)
    return next_node

  def backup(self, node, result):
    while node != None:
      player_in = self.switch_player(node.player)
      node.n_visit += 1
      node.value += result[player_in]
      node = node.parent


  def best_child(self, node, exploration=0):
    max_ = -100
    best = None
    for child in node.children:
      mean = child.value / child.n_visit
      explore = exploration * sqrt(2 * log(node.n_visit) / child.n_visit)
      ucb = mean + explore
      if ucb > max_:
        max_ = ucb
        best = child
    return best


  @staticmethod
  def switch_player(player):
    player = BLUE if player == YELLOW else YELLOW
    return player

if __name__ == '__main__':
  import numpy as np, sys
  node = Node(state=np.random.randn(9,20,20), player=1, parent=None, action_in=(0,0,0,0,0), terminal=False, result=None, depth=None)
  child = Node(state=np.random.randn(9,20,20), player=1, parent=node, action_in=(0,0,0,0,0), terminal=False, result=None, depth=None)
  print(sys.getsizeof(node))
  print(sys.getsizeof(child))



