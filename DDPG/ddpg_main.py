# THIS IS DDPG

import gym
import numpy as np
import torch
from ddpg_agent import Agent
from plot import plot_learning_curve
import os
from dotenv import load_dotenv

load_dotenv()
print("Using {} device".format(os.environ.get('DEVICE') if torch.cuda.is_available() else "cpu"))

# Hyperparams
n_games = 500
load_checkpoint = False
iterations = 3

env_id = 'LunarLanderContinuous-v2'
# env_id = 'BipedalWalker-v2'
# env_id = 'AntBulletEnv-v0'
# env_id = 'InvertedPendulumBulletEnv-v0'
# env_id = 'CartPoleContinuousBulletEnv-v0'

env = gym.make(env_id)
# if os.environ.get('RENDER') == "t":
#     env.render()
#     env.reset()

agent = Agent(alpha=0.0001, beta=0.001,
              input_dims=env.observation_space.shape, tau=0.001, env_id=env_id,
              batch_size=64, fc1_dims=400, fc2_dims=300,
              n_actions=env.action_space.shape[0])


def train(iteration):
    filename = agent.algo + "_" + env_id + "_" + str(n_games) + "games" + "_" + str(iteration) + '.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()
        if os.environ.get('RENDER') == "t":
            env.render(mode='human')

    steps = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        # for every episode:
        while not done:
            if os.environ.get('RENDER') == "t":
                env.render()

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(env_id, "|", 'episode', i, "|", 'score %.1f' % score, "|",
              '100 games avg %.1f' % avg_score, "|",
              'steps %d' % steps, "|"
              )

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file, agent.algo, env_id)


if __name__ == '__main__':
    for iteration in range(iterations):
        agent = Agent(alpha=0.0001, beta=0.001,
                      input_dims=env.observation_space.shape, tau=0.001, env_id=env_id,
                      batch_size=64, fc1_dims=400, fc2_dims=300,
                      n_actions=env.action_space.shape[0])
        train(iteration + 1)
