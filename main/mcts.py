import datetime
import multiprocessing
import os
import pickle
import random
import sys
import time
from copy import deepcopy

import numpy as np

# from concurrent.futures import ProcessPoolExecutor


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
    
    def work(self, state, root):
        t0 = time.time()
        iteration = 0
        np.random.seed()
        while time.time() - t0 < self.time_budget and iteration < self.iter_budget:
            leaf_node = self.tree_policy(root)

            if not leaf_node.terminal:
                result = self.default_policy(leaf_node)
            else:
                result = leaf_node.result
            self.backup(leaf_node, result)
            iteration += 1
            # print(f'\r__iteration: {iteration}__', end='')
        return root, iteration

    def get_action(self, state):
        """Assumes this function is NOT called on a terminal state"""
        time_budget = self.time_budget
        iter_budget = self.iter_budget
        print(f'Start: {datetime.datetime.now()} / Time budget: {time_budget} / Iteration budget: {iter_budget}')

        root = self.create_node(state)

        # root, iteration = self.work(state, root)

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            num_workers = pool._processes
            print(f'Running {num_workers} agent(s)!')
            forest = [pool.apply_async(self.work, (state, root)) for i in range(num_workers)]
            roots = [forest[i].get()[0] for i in range(num_workers)]
            iteration = np.sum([forest[i].get()[1] for i in range(num_workers)])
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

    def tree_policy(self, node):
        while not node.terminal:
            if len(node.children) < node.num_actions:
                # not fully expanded
                leaf = self.expand(node)
                return leaf

            else:
                # fully expanded
                player_in = node.player
                node = self.best_child(node, self.exploration)
        return node

    def default_policy(self, node):
        terminal = node.terminal
        assert not terminal
        state = node.state


        while not terminal:
            t_start = time.time()
            actions = self.env.possible_actions(state)
            t_action = time.time()
            action = random.choice(actions)
            state, result, terminal, _ = self.env.step(state, action, actions)
            t_step = time.time()

            player = state.board[self.env.TURN, 0, 0]
        return result

    def expand(self, node):
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        next_state, reward, done, _ = self.env.step(node.state, action, node.actions)
        if done:
            actions = []
        else:
            actions = self.env.possible_actions(next_state)

        # player = switch_player(node.player, self.player_list)
        kwargs = dict(
            state=next_state,
            parent=node,
            action_in=action,
            untried_actions=actions,
            terminal=done,
            result=reward if done else None
          )

        next_node = Node(**kwargs)
        # self.register_node(next_node)
        main_board = next_state.board[:21] + 2 * next_state.board[21:42]
        node.children[hash(main_board.tostring())] = next_node
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


    def get_node(self, state, player):
        try:
            raise KeyError('Temporary')
            main_board = state.board[:21] + 2 * state.board[21:42]
            node = self.nodes[hash(main_board.tostring())]
            node.parent = None
            print("Returning the requested Node...")
            return node

        except KeyError:
            node = self.create_node(state)
            print("Couldn't find the requested Node. Creating a new one...")
            return node

    def register_node(self, node):
        raise KeyError("Temporary")
        main_board = node.state.board[:21] + 2 * node.state.board[21:42]
        self.nodes[hash(main_board.tostring())] = node

    def create_node(self, state):
        actions = self.env.possible_actions(state)
        node = Node(state, parent=None, action_in=None, untried_actions=actions, cur_root=True)
        # self.register_node(node)
        return node

    def combine(self, roots):
        root = roots[0]
        for i in range(1, len(roots) - 1):
            root += roots[i]
        return root


class Node:
    def __init__(self, state, parent, action_in, untried_actions, terminal=False, result=None, cur_root=False):
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

        self.player = state.board[-1, 0, 0]
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
