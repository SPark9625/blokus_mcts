import sys, os
import time, datetime
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor

from copy import deepcopy
import pickle

BLUE, YELLOW = 1, 2

class UCT:
  def __init__(self, env, player_list, time_budget, num_workers, iter_budget=1600, exploration=1.4):
    self.env = env
    self.player_list = player_list
    assert len(self.player_list) == 2  # two player game for now

    self.time_budget = time_budget  # seconds
    self.iter_budget = iter_budget
    
    self.exploration = exploration
    self.nodes = {}

    self.num_workers = num_workers

  def get_action(self, state):
    """Assumes this function is NOT called on a terminal state"""
    time_budget = self.time_budget
    iter_budget = self.iter_budget
    print(f'Start: {datetime.datetime.now()} / Time budget: {time_budget} / Iteration budget: {iter_budget}')

    # t0 = time.time()
    # iteration = 0
    # root = self.create_node(state, state.player)
    # while time.time() - t0 < time_budget and iteration < iter_budget:
    #   remaining_pieces_all = deepcopy(state.remaining_pieces_all)
    #   leaf_node = self.tree_policy(root, remaining_pieces_all)
      
    #   if not leaf_node.terminal:
    #     result = self.default_policy(leaf_node, remaining_pieces_all)
    #   else:
    #     result = leaf_node.result
    #   self.backup(leaf_node, result)
    #   iteration += 1
    #   print(f'\r__iteration: {iteration}__', end='')
    
    with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
      num_workers = executor._max_workers
      print(f'Running {num_workers} agent(s)!')
      forest = []
      for i in range(num_workers):
        future = executor.submit(self.work, state)
        forest.append(future)
    roots = [forest[i].result()[0] for i in range(num_workers)]
    iteration = np.sum([forest[i].result()[1] for i in range(num_workers)])
    root = self.combine(roots)

    print('\n')
    children = sorted(root.children.values(), key=lambda x:x.mean, reverse=True)
    for child in children[:5]:
      print(child)
    print(f'End: {datetime.datetime.now()}')
    print(f'Finished {iteration} iterations.')


    best_node = self.best_child(root)
    action = best_node.action_in
    root.cur_root = False
    return action

  def tree_policy(self, node, remaining_pieces_all):
    while not node.terminal:
      if len(node.children) < node.num_actions:
        # not fully expanded
        leaf = self.expand(node, remaining_pieces_all)
        return leaf

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
    

    while not terminal:
      t_start = time.time()
      actions = self.env.possible_actions(state, player)
      t_action = time.time()
      action = random.choice(actions)
      state, result, terminal, _ = self.env.step(state, action, actions)
      t_step = time.time()

      remaining_pieces_all[player].remove(action[0])
      player = state.player
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
        terminal=done,
        result=reward if done else None
      )
    
    next_node = Node(**kwargs)
    # self.register_node(next_node)
    node.children[hash(next_state.board[0].tostring())] = next_node
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
    for child in node.children.values():
      ucb = child.mean + exploration * np.sqrt(2 * np.log(node.n_visit) / child.n_visit)
      if ucb > max_:
        max_ = ucb
        best = child
    return best


  def get_node(self, s0, player):
    try:
      raise KeyError('Temporary')
      node = self.nodes[hash(s0.board[0].tostring())]
      node.parent = None
      print("Returning the requested Node...")
      return node

    except KeyError:
      node = self.create_node(s0, player)
      print("Couldn't find the requested Node. Creating a new one...")
      return node

  def register_node(self, node):
    raise KeyError("Temporary")
    self.nodes[node.hash(state.board[0].tostring())] = node

  def create_node(self, s0, player):
    actions = self.env.possible_actions(s0, player)
    node = Node(s0, player=player, parent=None, action_in=None, untried_actions=actions, cur_root=True)
    # self.register_node(node)
    return node

  def combine(self, roots):
    root = roots[0]
    for i in range(1, len(roots) - 1):
      root += roots[i]
    return root

  def work(self, state):
    root = self.create_node(state, state.player)

    t0 = time.time()
    iteration = 0
    while time.time() - t0 < self.time_budget and iteration < self.iter_budget:
      remaining_pieces_all = deepcopy(state.remaining_pieces_all)
      leaf_node = self.tree_policy(root, remaining_pieces_all)
      
      if not leaf_node.terminal:
        result = self.default_policy(leaf_node, remaining_pieces_all)
      else:
        result = leaf_node.result
      self.backup(leaf_node, result)
      iteration += 1
      # print(f'__iteration: {iteration}__{os.getpid()}')
    return root, iteration


class Node:
  def __init__(self, state, player, parent, action_in, untried_actions, terminal=False, result=None, cur_root=False):
    # distinguishing feature
    self.state = state

    self.parent = parent
    self.action_in = action_in
    self.actions = deepcopy(untried_actions)
    self.untried_actions = deepcopy(untried_actions)  # variable
    self.num_actions = len(untried_actions)

    self.children = {}

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
    print(f'Node:\n\taction_in: {self.action_in},\n\tnum_children: {len(self.children)},\
\n\tterminal: {self.terminal},\n\tresult: {self.result},\n\tplayer: {self.player}\n\tstate:\n{self.state[0]}')

  def __repr__(self):
    return f'{self.action_in}: {self.value}/{self.n_visit} (={str(self.mean)[:5]})'

  def __add__(self, node):
    keys = set(self.children.keys()) | set(node.children.keys())
    for key in keys:
      this_child = self.children.get(key, None)
      that_child = node.children.get(key, None)
      if this_child == None:
        self.children[key] = that_child
      elif that_child == None:
        pass
      else:
        # both have the same child
        this_child.value += that_child.value
        this_child.n_visit += that_child.n_visit
    return self
    


