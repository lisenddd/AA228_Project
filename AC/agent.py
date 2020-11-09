import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

class Agent():
    
    def __init__(self, state_size, action_size, lr, gamma, device='cpu',selfplay=False,loadold=False):
        # Save parameter in class
        self.lr     = lr                # Learning rate
        self.gamma  = gamma             # Discounted reward factor
        self.device = device            # Device to execute
        self.state_size  = state_size   # State size
        self.action_size = action_size  # Action size

        if loadold:
            self.loadNetwork()
        else:
            self.actorcritic      = A2C(self.state_size, self.action_size, self.lr, self.device)
            if selfplay:
                self.actorcritic_targ = A2C(self.state_size, self.action_size, self.lr, self.device)

    def take_action(self,s):
        state = torch.tensor(s)

        _, action_probs = self.actorcritic.forward(state)
        action_dist = Categorical(action_probs)
        
        # Pick action | Exploration achieved by sampling
        action = action_dist.sample()
        self.action = action
        return action.cpu().numpy()

    def take_action2(self,s):
        state = torch.tensor(s)

        _, action_probs = self.actorcritic_targ.forward(state)
        action_dist = Categorical(action_probs)
        
        # Pick action | Exploration achieved by sampling
        action = action_dist.sample()
        # self.action = action
        return action.cpu().numpy()

    def saveNetwork(self):
        torch.save(self.actorcritic, 'model/ACnet.pkl')
        torch.save(self.actorcritic_targ, 'model/ACnet_targ.pkl')
        print('Saved network')

    def loadNetwork(self):
        self.actorcritic = torch.load('model/ACnet.pkl').to(self.device)
        self.actorcritic_targ = torch.load('model/ACnet_targ.pkl').to(self.device)
        print('Model loaded')

    def update_target(self,targetnet,sourcenet):
        for target_param, source_param in zip(targetnet.parameters(), sourcenet.parameters()):
            target_param.data.copy_(source_param.data)
        # targetnet.load_state_dict(sourcenet.state_dict())
        print('Copied network')

    def takeaction(self,s):
        state = torch.tensor(s)

        _, action_probs = self.actorcritic.forward(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        self.action = action

        return action.numpy()

    def train(self,s,r,sp,done):
        # Convert to tensor
        state = torch.tensor(s)
        reward = torch.tensor(r)
        new_state = torch.tensor(sp)

        # Clear gradient
        self.actorcritic.optim.zero_grad()
                
        # Get value estimation for current state(value0) and next state(value1)
        critic_value0,action_probs = self.actorcritic.forward(state)
        critic_value1,_     = self.actorcritic.forward(new_state)
        
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(self.action)

        # TD error calculation
        delta = reward + self.gamma*critic_value1*(1-int(done)) - critic_value0

        # Find loss of actor and critic
            # AC loss
        actor_loss  = - log_probs * delta
        critic_loss = delta**2
        AC_loss = actor_loss + critic_loss

        # Propagate loss and update net
        AC_loss.backward()
        self.actorcritic.optim.step()

        lossval = AC_loss.data.item()
        return lossval


class A2C(nn.Module):
    def __init__(self, state_dim, action_dim, lr, device='cpu'):
        super().__init__()

        # Save parameter to class
        self.device     = device
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Layer definitions
        self.lin1 = nn.Linear(512,128)
        self.lin2 = nn.Linear(128,512)
        self.conv1 = nn.Conv2d(1,32,kernel_size=6,stride=1)
        self.fc_c  = nn.Linear(512,1)
        self.fc_a  = nn.Linear(512,action_dim)

        # Optimizer definition
        self.optim = torch.optim.Adam(self.parameters(),lr)

        self.to(self.device)

    def forward(self, s):
        # Convert state to tensor with correct shape and device
        # state = torch.tensor(s.flatten()).to(self.device).float() # flatten state
        state = torch.tensor(s).to(self.device).float().unsqueeze(0).unsqueeze(0)           # square state
        
        # Connect Layers
        # Specifically, the layers are constructed this way:
        # 
        #        x1           x2 .--> fc_c (critic: value)
        # Linear1 -----> Linear2--
        #                        `--> fc_a (actor : action)

        # x1 = F.relu(self.lin1(state.flatten()))
        # x2 = F.relu(self.lin2(x1))

        x0 = F.relu(self.conv1(state))
        x1 = F.relu(self.lin1(x0.view(-1)))
        x2 = F.relu(self.lin2(x1))
        
        value  = self.fc_c(torch.flatten(x2))
        actnet_output = self.fc_a(torch.flatten(x2))
        actnet_output[s.flatten()>0] = -1000000000.0
        # print(actnet_output)
        action = F.softmax(actnet_output)

        return value, action
