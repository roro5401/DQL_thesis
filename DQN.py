import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd

"""
This class creates a Deep Q-Network, which is used for decision making by the agent. The following functions are
included:

1. __init__(): initiliazes the DQN
2. forward(): propogates an object through the DQN to retreive an output
3. policy(): returns the best known action
"""

class DQN(nn.Module):

    """
    Initializes the DQN.

    @param input_size: the size of the inputted state vector.
    @param output_size: the possible number of actions for the agent.
    """

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.input = input_size
        self.lin1 = nn.Linear(in_features=input_size, out_features = 80)
        self.lin2 = nn.Linear(in_features=80, out_features=100)
        self.lin3 = nn.Linear(in_features=100, out_features=75)
        self.out = nn.Linear(in_features=75, out_features=output_size)
        self.name = "DQN"


    """
    Passes the object x through the DQN and returns the last hidden state.

    @param x: input state for the DQN.
    @return x: the last hidden state of the DQN.
    """
    def forward(self, x):
        x = torch.FloatTensor(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.out(x))
        return x

    """
    Passes the object x through the DQN and returns the index of the highest scoring action.

    @param x: input state for the DQN.
    @return x: the index of the highest scoring action.
    """
    def policy(self, state):
        x = self.forward(state)
        with torch.no_grad():
            return x.argmax(dim=0)

"""
Creates a named tuple containing current state, action, next state, and reward. 
Used to store experiences in the replay memory.
"""
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


"""
This class creates a Replay Memory in which experiences are stored to train the DQN.The following functions are
included:

1. __init__(): initiliazes the Replay Memory
2. push(): adds an experience to the Replay Memory
3. sample(): takes a random sample from the Replay Memory
4. can_provide_sample(): returns a boolean, true if sample can be provided, false otherwise,
"""
class ReplayMemory():
    """
    Initializes the Replay Memory.

    @param capacity: the maximum number of experiences that are stored
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    """
    Adds an experience to the memory.

    @param experience: experience tuple containing current state, action, next state, and reward.
    """
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    """"
    Provides a sample from the memory.
    
    @param batch_size: size of the sample
    @return sample from replay memory
    """
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    """"
    Checks whether a sample can be provided.

    @param batch_size: size of the sample
    @return boolean, true if sample is availanle, false otherwise
    """
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

"""
This class creates an Epsilon Greedy Strategy. This strategy is used to create the exploration-exploitation balance.
The following functions are available:

1. __init__(): initializes the EpsilonGreedyStrategy
2. get_exploration_rate(): returns the current exploration rate
"""
class EpsilonGreedyStrategy():
    """
    Initializes an instance of the EpsilonGreedyStrategy()

    @param start: starting exploration rate
    @param end: minimum value of the exploration rate
    @param decay: decay rate with each current step
    """
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    """
    Returns the exploration rate for the current step. The used strategy is exponential decay.
    
    @param current_step: current step in the exploration scheme.
    """

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

"""
This class creates an Agent thaat is able to interact with the environment. The following functions are available:

1. __init__(): initializes the Agent
2. training_policy(): returns the action according to the training policy.
3. policy(): returns the best available action.
4. update_step(): updates the current training step of the agent.
"""
class Agent():

    """
    Initializes an instance of the Agent()

    @param strategy: EpsilonGreedyStrategy for training the agent.
    @param num_actions: number of available actions.
    @param critical_maintenance_ages: p-ARP policy injected into the agent for decision making. For more information,
    see section 4.4.2 of the paper.
    @param device: device on which the computations for the agent are run.
    @param policy_path: the directory path to load in a pretrained agent's policy network.
    """
    def __init__(self, strategy, num_actions, critical_maintenance_ages, device, policy_path = None):
        if policy_path is not None:
            self.policy_net = torch.load(policy_path)
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.cma = critical_maintenance_ages
        self.name = "DQN"

    """
    Selects an action throughout training. It selects a random action with probability equal to the 
    EpsilonGreedyStrategy. Otherwise, it selects its best known action.
    
    @param state: input for the policy net, the expected state at the starte of period t+1.
    @param policy_net: the agent's policy network, instance of a DQN class.
    @return the number of scheduled PMs for period t+1
    """
    def training_policy(self, state, policy_net):
        time = state[0] - 2
        rate = self.strategy.get_exploration_rate(self.current_step)
        lb = max(len([i for i in state[1:] if i >= self.cma[time] - self.experiment_margin]), 0)
        ub = self.num_actions - state[1:].count(0) - 1

        if rate > random.random():
            action = random.randrange(lb, ub+1)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state)[lb:].argmax(dim=0).to(self.device)

    """
    Selects the best known action. This function is used in comparing the policies in simulations.py
    
    @param state: input for the policy net, the expected state at the starte of period t+1.
    @return the number of scheduled PMs for period t+1, the best known action.
    """
    def policy(self, state):
        time = state[0] - 2
        lb = max(0, len([i for i in state[1:] if i >= self.cma[time]]) - self.experiment_margin)
        with torch.no_grad():
            action = self.policy_net(state)[lb:].argmax(dim = 0).to(self.device)
            return action.item()

    """
    Updates the current step of the training procedure. Necessary for the epsilon greedy strategy.
    """
    def update_step(self):
        self.current_step += 1

"""
This class is used to estimate Q-values according to the procedure described in the paper, section 4.4.
The following functions are available:

1. __init__(): initializes the Agent
2. training_policy(): returns the action according to the training policy.
3. policy(): returns the best available action.
4. update_step(): updates the current training step of the agent.
"""
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.long().unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        return target_net(next_states).max(dim=1)[0].detach()

"""
This function is used to extract a batch from the replay memory and turning it into tensors. This is necessary to train
the PyTorch model.

@return tuple of tensors required for Q-value estimation and policy network training.
"""
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.FloatTensor(batch.state)
    t2 = torch.FloatTensor(batch.action)
    t3 = torch.FloatTensor(batch.reward)
    t4 = torch.FloatTensor(batch.next_state)

    return (t1, t2, t3, t4)
