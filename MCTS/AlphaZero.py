import numpy as np
import random
from AlphaZero_agent import ZeroPlayer
from utils import PolicyValueNet
import wandb
import gym
from collections import deque
from gym_gomoku.envs.util import gomoku_util
from copy import deepcopy
import torch


class AlphaZeroTrainer:
    def __init__(self, trained_model=None):
        self.env = gym.make('Gomoku7x7-v0')
        self.board_width = 7
        # training params
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_param = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 2000
        self.best_win_ratio = 0.0
        # # num of simulations used for the pure mcts, which is used as
        # # the opponent to evaluate the trained policy
        # self.pure_mcts_playout_num = 1000
        self.policy_value_net = PolicyValueNet(self.board_width, trained_model)
        self.zero_player = ZeroPlayer(self.policy_value_net.policy_value_fn, self.c_param, self.n_playout, True)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.rot90(state, i)
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_width, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.fliplr(state)
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def self_play(self):
        player_map = {'black': 1, 'white': 2, 'empty': -1}
        _, state = self.env.reset()
        states, mcts_probs, current_players = [], [], []
        while True:
            state = deepcopy(state)
            move, move_probs = self.zero_player.get_action(state, self.temp, True)
            states.append(state.board.board_state)
            mcts_probs.append(move_probs)
            current_players.append(player_map[state.color])
            state = state.act(move)

            if not state.board.is_terminal():
                oppo_move, oppo_move_probs = self.zero_player.get_action(state, self.temp, True)
                states.append(state.board.board_state)
                mcts_probs.append(oppo_move_probs)
                current_players.append(player_map[state.color])
                observation, reward, done, state = self.env.step(move, oppo_move)
            else:
                observation, reward, done, state = self.env.step(move)

            _, winner = gomoku_util.check_five_in_row(state.board.board_state)
            winner = player_map[winner]
            if done:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                self.zero_player.reset_player()
                _, state = self.env.reset()
                return winner, zip(states, mcts_probs, winners_z)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        wins = []
        for i in range(n_games):
            winner, play_data = self.self_play()
            wins.append(winner)
            play_data = list(play_data)[:]
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch]).reshape((-1, 1, 7, 7))
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch])
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss = self.policy_value_net.train(state_batch,
                                               mcts_probs_batch,
                                               winner_batch,
                                               self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               ).format(kl,
                        self.lr_multiplier,
                        loss))
        wandb.log({"kl": kl, "lr multiplier": self.lr_multiplier, "loss": loss})
        return loss

    def policy_evaluate(self, n_games=10):
        zero_player = ZeroPlayer(self.policy_value_net.policy_value_fn, self.c_param, self.n_playout)
        rewards = 0
        _, state = self.env.reset()
        for i in range(n_games):
            done = False
            while not done:
                action = zero_player.get_action(state)
                observation, reward, done, state = self.env.step(action)
                if i == 9:
                    self.env.render("human")
            zero_player.reset_player()
            rewards += reward if reward > 0 else 0
            _, state = self.env.reset()
        win_rate = rewards / n_games
        wandb.log({"win rate": win_rate})
        return win_rate

    def run(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        wandb.init()
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            if len(self.data_buffer) > self.batch_size:
                self.policy_update()
            # check the performance of the current model,
            # and save the model params
            if (i + 1) % self.check_freq == 0:
                print("current self-play batch: {}".format(i + 1))
                win_ratio = self.policy_evaluate()
                self.policy_value_net.save_model('./current_policy.model')
                if win_ratio > self.best_win_ratio:
                    print("New best policy!")
                    self.best_win_ratio = win_ratio
                    # update the best_policy
                    self.policy_value_net.save_model('./best_policy.model')


if __name__ == '__main__':
    trainer = AlphaZeroTrainer()
    trainer.run()
