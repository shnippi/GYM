# THIS IS DDPG

import gym
import numpy as np
import torch
from lunar_agent import Agent
# from plot import *
import os
from dotenv import load_dotenv

# Action is two real values vector from -1 to +1. First controls main engine, -1...0 off, 0...+1
# throttle from 50% to 100% power. Engine can't work with less than 50% power. Second value -1.0..-0.5 fire left engine,
# +0.5..+1.0 fire right engine, -0.5..0.5 off.

load_dotenv()

if __name__ == '__main__':
    device = os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"
    print(print("Using {} device".format(device)))

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            # env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
    x = [i + 1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)
