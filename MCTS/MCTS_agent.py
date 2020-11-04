import numpy as np
import copy
import gym
from gym_gomoku.envs.util import gomoku_util


class Node(object):
    def __init__(self, parent):
        self._parent = parent
        self._children = {}  # {action1:Node, action2:Node, ...}
        self._n_visits = 0
        self._value = 0  # state value
        self._u = 0  # exploration value

    def select(self, c_param):
        """
        Greedy selection base on the total score of a node
        c_param is the exploration parameters controls the weight of exploration value in total score
        Return the child node with highest sum of state value and exploration value
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_param))

    def expand(self, actions):
        # expand node with all legal actions
        for action in actions:
            if action not in self._children:
                self._children[action] = Node(self)

    def update(self, leaf_value):
        """
        leaf_value: the value of subtree evaluation from the current player's perspective.
        """
        self._n_visits += 1
        self._value += (leaf_value - self._value) / self._n_visits  # running average

    def update_recursive(self, leaf_value):
        # If self is not root, update its parent first
        if self._parent:
            self._parent.update_recursive(-leaf_value)  # every level the player switches, so there's a negative sign
        self.update(leaf_value)

    def get_value(self, c_param):
        """
        Return the UCB score of current node
        """
        self._u = c_param * np.sqrt(np.log(self._parent._n_visits) / (1 + self._n_visits))
        return self._value + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, c_param=2, n_playout=10000):
        self._root = Node(None)
        self._c_param = c_param
        self._n_playout = n_playout

    def _playout(self, state):
        """
        Single rollout of a leaf node
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_param)
            state = state.act(action)

        if not state.board.is_terminal():
            node.expand(state.board.get_legal_action())
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """
        Return 1 if current player wins, -1 if other player wins, 0 for tie
        """
        player = state.color
        for i in range(limit):
            if state.board.is_terminal():
                break
            state = state.act(np.random.choice(state.board.get_legal_action()))
        else:
            print("WARNING: rollout reached move limit")

        exist, win_color = gomoku_util.check_five_in_row(state.board.board_state)
        if win_color not in ['black', 'white']:
            return 0
        return 1 if win_color == player else -1

    def get_move(self, state):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def reset_root(self):
        self._root = Node(None)


def main():
    MCTS_agent = MCTS()
    env = gym.make('Gomoku7x7-v0')
    _, state = env.reset()
    done = False
    while not done:
        env.render('human')
        action = MCTS_agent.get_move(state)
        MCTS_agent.reset_root()
        observation, reward, done, state = env.step(action)
    env.render('human')
    print(reward)


if __name__ == '__main__':
    main()
