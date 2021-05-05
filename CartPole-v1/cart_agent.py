import torch
import random
import numpy as np
from collections import deque
from cart_model import Linear_QNet, QTrainer
import os
from dotenv import load_dotenv

load_dotenv()

# Get cpu or gpu device for training.
device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01


class Agent:

    def __init__(self, gamma, epsilon, n_actions):
        self.n_games = 0
        self.epsilon = epsilon
        self.eps_min = 0.01
        self.eps_dec = 0.005
        self.gamma = gamma  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(4, 256, 2).to(
            device)  # 4 value state input, 2 action as output 0/1
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.action_space = [i for i in range(n_actions)]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    # after every episode
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # after every action
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation, shrinking epsilon
        if np.random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        return action
