# THIS IS TD3

import gym
import numpy as np
from td3_agent import Agent
from plot import plot_learning_curve
import torch
from dotenv import load_dotenv
import os
load_dotenv()
print("Using {} device".format(os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"))

# Hyperparams
n_games = 500
load_checkpoint = False


# env_id = 'BipedalWalker-v3'
env_id = 'LunarLanderContinuous-v2'

env = gym.make(env_id)
agent = Agent(alpha=0.001, beta=0.001,
              input_dims=env.observation_space.shape, tau=0.005,
              env=env, env_id=env_id, batch_size=100, layer1_size=400, layer2_size=300,
              n_actions=env.action_space.shape[0])



