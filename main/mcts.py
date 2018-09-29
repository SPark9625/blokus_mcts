import sys
import time, datetime
import numpy as np
import random
from copy import deepcopy
import pickle


BLUE, YELLOW = 1, 2
# 1. 노드 에 actions를 달아서 tree policy할 때 빠르게.
# => 이거 하려면 state에 player 넣어도 될듯. 그러면 노드 생성할때도 그냥 player 그대로 넣으면 되고, switch_player같은 함수는 다 없애도됨.
# 2. 

class Node:
  def __init__(self, state, player, parent, action_in, untried_actions, terminal=False, result=None, depth=None, cur_root=False):
    self.depth = depth
    self.state = state
    self.parent = parent
    self.action_in = action_in
    self.actions = deepcopy(untried_actions)
    self.untried_actions = deepcopy(untried_actions)
    self.num_actions = len(untried_actions)

    self.children = []

    self.value = 0
    self.n_visit = 0

    self.terminal = terminal
    self.result = result

    self.player = player
    self.cur_root = cur_root

  @property
  def mean(self):
    return self.value/(self.n_visit + 1e-10)

  def print(self):
    print(f'Node {self.depth}:\n\taction_in: {self.action_in},\n\tnum_children: {len(self.children)},\
\n\tterminal: {self.terminal},\n\tresult: {self.result},\n\tplayer: {self.player}\n\tstate:\n{self.state[0]}')

  def __repr__(self):
    return f'{self.action_in}: {self.value}/{self.n_visit} (={str(self.mean)[:5]})'

    


class UCT:
  def __init__(self, env, player, player_list, time_budget, iter_budget=1600, exploration=1.4):
    """env has to have 2 APIs: env.step() and env.possible_actions()
      env.step() takes (state, action, player) as input and has to return: (next_state, reward, done, info)
      env.possible_actions() takes `state`, `player_id`, `piece_idx` as input and has to return a `list` of actions.
    """
    self.env = env
    self.IAM = player
    self.player_list = player_list
    self.opponent = player_list[0] if player == player_list[1] else player_list[1]
    assert len(self.player_list) == 2  # two player game for now

    self.time_budget = time_budget  # seconds
    self.iter_budget = iter_budget
    
    self.exploration = exploration
    self.nodes = {}
    ### TODEL
    self.first = True


  def get_action(self, state):
    """Assumes this function is NOT called on a terminal state"""
    time_budget = self.time_budget
    iter_budget = self.iter_budget
    if self.first:
      time_budget = 20
      self.first = False

    print(f'Start: {datetime.datetime.now()} / Time budget: {time_budget} / Iteration budget: {iter_budget}')
    root = self.get_node(state, self.IAM)
    root.cur_root = True
    iteration = 0

    
    t0 = time.time()
    global time_default
    time_default = {'action':[], 'step':[], 'total_def':[], 'total_tree':[]}


    while time.time() - t0 < time_budget and iteration < iter_budget:
      remaining_pieces_all = deepcopy(state.remaining_pieces_all)
      
      t_start = time.time()
      
      leaf_node = self.tree_policy(root, remaining_pieces_all)
      
      t_tree = time.time()
      time_default['total_tree'].append(t_tree - t_start)
      
      if not leaf_node.terminal:
        t_start = time.time()
        
        result = self.default_policy(leaf_node, remaining_pieces_all)
        
        t_def = time.time()
        time_default['total_def'].append(t_def - t_start)
      else:
        result = leaf_node.result
      self.backup(leaf_node, result)
      iteration += 1
      sys.stdout.write(f'\r__iteration: {iteration}__')
      sys.stdout.flush()
      

    print('\n')
    root.children.sort(key=lambda x:x.mean, reverse=True)
    for child in root.children[:5]:
      print(child)
    print(f'Finished {iteration} iterations.')


    print('total_tree:', np.sum(time_default['total_tree']))
    print('total_def:', np.sum(time_default['total_def']))
    print('  action:', np.sum(time_default['action']))
    print('  step:', np.sum(time_default['step']))
    

    best_node = self.best_child(root)
    action = best_node.action_in
    root.cur_root = False
    return action

  def tree_policy(self, node, remaining_pieces_all):
    while not node.terminal:
      if len(node.children) < node.num_actions:
        # not fully expanded
        return self.expand(node, remaining_pieces_all)

      else:
        # fully expanded
        player_in = node.player
        node = self.best_child(node, self.exploration)
        remaining_pieces_all[player_in].remove(node.action_in[0])
        

    return node

  def default_policy(self, node, remaining_pieces_all):
    terminal = node.terminal
    assert not terminal
    state = node.state
    player = node.player
    

    global time_default
    while not terminal:
      t_start = time.time()
      actions = self.env.possible_actions(state, player)
      t_action = time.time()
      # if len(actions) == 0:
      #   # print(f'player {player} has no more actions')
      #   player = switch_player(player, self.player_list)
      #   continue
      action = random.choice(actions)
      state, result, terminal, _ = self.env.step(state, action, actions)
      t_step = time.time()

      remaining_pieces_all[player].remove(action[0])
      player = state.player
      time_default['action'].append(t_action - t_start)
      time_default['step'].append(t_step - t_action)
    return result

  def expand(self, node, remaining_pieces_all):
    action = random.choice(node.untried_actions)
    node.untried_actions.remove(action)
    remaining_pieces_all[node.player].remove(action[0])

    next_state, reward, done, _ = self.env.step(node.state, action, node.actions)
    if done:
      actions = []
    else:
      actions = self.env.possible_actions(next_state, next_state.player)

    # player = switch_player(node.player, self.player_list)
    kwargs = dict(
        state=next_state,
        player=next_state.player,
        parent=node,
        action_in=action,
        untried_actions=actions,
        depth=node.depth + 1,
        terminal=done,
        result=reward if done else None
      )
    
    next_node = Node(**kwargs)
    self.register_node(next_node)
    node.children.append(next_node)
    return next_node

  def backup(self, node, result):
    while not node.cur_root:
      player_in = node.parent.player
      node.n_visit += 1
      node.value += result[player_in]
      node = node.parent
    # update root's visit count.
    node.n_visit += 1
    node.value += result[node.player]


  def best_child(self, node, exploration=0):
    max_ = -100
    best = None
    for child in node.children:
      ucb = child.mean + exploration * np.sqrt(2 * np.log(node.n_visit) / child.n_visit)
      if ucb > max_:
        max_ = ucb
        best = child
    return best


  def get_node(self, s0, player):
    try:
      node = self.nodes[s0.board[0].tobytes()]
      node.parent = None
      print("Returning the requested Node...")
      return node

    except KeyError:
      actions = self.env.possible_actions(s0, player)
      node = Node(s0, player=player, parent=None, action_in=None, untried_actions=actions, depth=0)
      self.register_node(node)
      print("Couldn't find the requested Node. Creating a new one...")
      return node

  def register_node(self, node):
    self.nodes[node.state.board[0].tobytes()] = node




