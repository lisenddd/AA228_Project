import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Model(nn.Module):
    def __init__(self, board_width):
        super(Model, self).__init__()
        self.board_size = board_width ** 2
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 64, 3)

        self.action_fc = nn.Linear(64, 128)
        self.action = nn.Linear(128, self.board_size)
        self.value_fc = nn.Linear(64, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        action = F.relu(self.action_fc(x.view(-1, 64)))
        action = F.log_softmax(self.action(action), dim=1)

        value = F.relu(self.value_fc(x.view(-1, 64)))
        value = torch.tanh(self.value(value))
        return action, value


class PolicyValueNet:
    def __init__(self, board_width, trained_model=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.board_width = board_width
        self.policy_value_net = Model(board_width).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-5)
        if trained_model:
            self.policy_value_net.load_state_dict(torch.load(trained_model))

    def policy_value(self, state_batch):
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, state):
        legal_moves = state.board.get_legal_action()
        board_state = np.reshape(state.board.board_state, (-1, 1, self.board_width, self.board_width))
        log_act_probs, value = self.policy_value_net(torch.from_numpy(board_state).float().to(self.device).float())
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        act_probs = zip(legal_moves, act_probs[legal_moves])
        return act_probs, value.item()

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        mcts_probs = torch.from_numpy(mcts_probs).float().to(self.device)
        winner_batch = torch.from_numpy(winner_batch).float().to(self.device)

        self.optimizer.zero_grad()
        self.set_learning_rate(lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
