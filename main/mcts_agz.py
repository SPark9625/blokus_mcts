import datetime
import os
import random
from copy import deepcopy
from math import log, sqrt
from time import sleep, time

import numpy as np
from torch.multiprocessing import Pool

from network import PVNet as tf_net
from network_pt import PVNet as pt_net
from util import softmax
from config import config


class MCTS:
    def __init__(self, env, tau, c_puct=4):
        self.nodes = {}

        # env related
        self.env         = env
        self.player_list = env.player_list

        if config.framework in ['tf','tensorflow']:
            self.model = tf_net()
        elif config.framework in ['pt','pytorch']:
            self.model = pt_net(env.BOARD_SHAPE, env.ACTION_SHAPE)
        else:
            raise ValueError('Choose from `tf` / `pt`. Review `./config.py`')
        self.data_format = self.model.data_format

        # constants
        self.tau         = tau
        self.c_puct      = c_puct
        self.alpha       = 10 / config.board_size ** 2
        self.time_budget = config.time_budget
        self.iter_budget = config.iter_budget
        self.num_workers = config.num_workers

    def simulate(self, root):
        log = {}
        log['selection'] = {'num': 0, 'time': 0}
        log['exp'] = {'num': 0, 'time': 0}
        log['backup'] = {'num': 0, 'time': 0}
        # for expansion/eval
        self.log_ = {}
        self.log_['exp/step'] = {'num': 0, 'time': 0}
        self.log_['exp/eval'] = {'num': 0, 'time': 0}

        print('Start:', str(datetime.datetime.now()))
        iteration = 0
        start = time()
        while (time() - start < self.time_budget and iteration < self.iter_budget) or iteration < root.num_actions:
            time_s = time()
            leaf = self.select(root)
            log['selection']['time'] += (time() - time_s); log['selection']['num'] += 1
            if not leaf.terminal:
                time_e = time()
                V, child = self.expand_and_eval(leaf)  # this expands and evaluates
                log['exp']['time'] += (time() - time_e); log['exp']['num'] += 1
                time_b = time()
                self.backup(V, child)
                log['backup']['time'] += (time() - time_b); log['backup']['num'] += 1
            else:
                time_b = time()
                V = leaf.reward
                self.backup(V, leaf)
                log['backup']['time'] += (time() - time_b); log['backup']['num'] += 1
            iteration += 1
            print(f'\riteration: {iteration} | pid: {os.getpid()}', end='')
        print()
        print('End  :', str(datetime.datetime.now()))
        print('Took :', str(datetime.timedelta(seconds=time() - start)))
        print('selection:', log['selection'])
        print('backup   :', log['backup'])
        print('expansion:', log['exp'])
        print('    step :', self.log_['exp/step'])
        print('    eval :', self.log_['exp/eval'])
        print()
        return root, iteration

    def get_action(self, state):
        root = self.get_root(state)
        
        self.simulate(root)

        best_child = self.best(root, final=True)
        return best_child.action_in

    def get_root(self, state):
        main_board = state.board[:self.env.DIAGONAL].sum(axis=0)
        key = hash(main_board.tostring())
        if key in self.nodes.keys():
            root = self.nodes[key]
            # This makes this node root
            root.parent = None
        else:
            # Create a node
            # At the root node, evaluation and backup occurs at creation
            actions = state.meta.actions[state.board[-1,0,0]]
            prior_raw, V = self.evaluate(state.board)
            prior = np.zeros_like(prior_raw)
            noise = np.random.dirichlet(np.ones(len(actions)) * self.alpha)
            i0, i1, i2 = np.array(actions).T
            if self.data_format == 'channels_last':
                i2, i0, i1 = i0, i1, i2

            prior[np.zeros_like(i0), i0, i1, i2] = softmax(
                prior_raw[np.zeros_like(i0), i0, i1, i2]) + noise
            kwargs = {
                'state'    : state,
                'p'        : None,
                'prior'    : prior,
                'parent'   : None,
                'action_in': None,
                'actions'  : actions,
            }
            root = Node(**kwargs)
            self.nodes[key] = root
            self.backup(V, root)
        return root
    
    def select(self, node):
        while node.fully_expanded and not node.terminal:
            node = self.best(node)
        return node

    def expand_and_eval(self, leaf):
        action = random.choice(leaf.untried)
        leaf.untried.remove(action)

        time_s = time()
        state, reward, done, _ = self.env.step(leaf.state, action, leaf.actions)
        self.log_['exp/step']['time'] += time() - time_s; self.log_['exp/step']['num'] += 1
        if done:
            actions = []
            prior = None
            V = reward
            reward = reward
        else:
            actions = state.meta.actions[state.board[-1, 0, 0]]
            time_e = time()
            prior_raw, V = self.evaluate(state.board)
            prior = np.zeros_like(prior_raw)
            noise = np.random.dirichlet(np.ones(len(actions)) * self.alpha)
            i0, i1, i2 = np.array(actions).T
            if self.data_format == 'channels_last':
                i0, i1, i2 = i1, i2, i0

            prior[np.zeros_like(i0), i0, i1, i2] = softmax(
                prior_raw[np.zeros_like(i0), i0, i1, i2]) + noise
            self.log_['exp/eval']['time'] += time() - time_e; self.log_['exp/eval']['num'] += 1
            reward = None

        if self.data_format == 'channels_last':
            p = leaf.prior[(0, action[1], action[2], action[0])]
        else:
            p = leaf.prior[(0, *action)]
        kwargs = {
            'state'    : state,
            'p'        : p,
            'prior'    : prior,
            'parent'   : leaf,
            'action_in': action,
            'actions'  : actions,
            'reward'   : reward,
            'terminal' : done,
        }
        child = Node(**kwargs)

        # Register node
        main_board = state.board[:self.env.DIAGONAL].sum(axis=0)
        key = hash(main_board.tostring())
        self.nodes[key] = child

        # Declar child of its parent
        leaf.children[action] = child
        return V, child

    def train(self):
        if self.model.framework in ['pt', 'pytorch']:
            self.model.train()
        return self
    
    def eval(self):
        if self.model.framework in ['pt', 'pytorch']:
            self.model.eval()
        return self

    def evaluate(self, board):
        prior, V = self.model(board)
        if self.model.framework in ['pt', 'pytorch']:
            prior = prior.cpu().detach().numpy()
            V = V.cpu().detach().numpy()
        return prior, V

    def backup(self, V, child):
        node = child
        while True:
            node.n += 1
            node.w += V[0, node.player]
            if node.parent == None:
                break
            node = node.parent
    
    def best(self, root, final=False):
        if final:
            max_visit = max([child.n for child in root.children.values()])
            max_children = [child for child in root.children.values() if child.n == max_visit]
            return random.choice(max_children)
        max_value = max([child.ucb(self.c_puct, root.n) for child in root.children.values()])
        max_children = [child for child in root.children.values() if child.ucb(self.c_puct, root.n) == max_value]
        return random.choice(max_children)
        


class Node:
    def __init__(self, state, p, prior, parent, action_in, actions, terminal=False, reward=None):
        # core
        self.state = state

        # statistics
        self.w = 0
        self.n = 0
        self.p = p

        # (s,a)
        self.parent    = parent
        self.action_in = action_in

        # other important info
        self.children = {a: None for a in actions}
        self.player   = state.board[-1,0,0]
        self.prior    = prior
        self.terminal = terminal
        self.reward   = reward  # this will have a value iff node is terminal

        # for efficiency
        self.actions     = actions
        self.untried     = deepcopy(actions)
        self.num_actions = len(actions)

    @property
    def q(self):
        return self.w / self.n

    def ucb(self, c_puct, parent_n):
        return self.q + c_puct * self.p * sqrt(parent_n) / (1 + self.n)

    @property
    def fully_expanded(self):
        if not self.untried:
            return True
        return False


if __name__ == '__main__':
    from model import Blokus
    env = Blokus()
    agent = MCTS(env, tau=1, c_puct=2)
    # state = env.reset()
    # agent.get_action(state)
