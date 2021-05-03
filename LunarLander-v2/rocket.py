import gym
import numpy as np

from rocket_agent import Agent

env = gym.make('LunarLander-v2')
n_actions = env.action_space.n
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=n_actions, eps_end=0.01, input_dims=[8], lr=0.003)
scores, eps_history = [], []
epochs = 500

for i_episode in range(epochs):
    score = 0
    done = False
    state_old = env.reset()
    while not done:  # iterating over every timestep (state)
        env.render()
        action = agent.choose_action(state_old)
        state_new, reward, done, info = env.step(action)
        score += reward

        agent.store_transition(state_old, action, reward, state_new, done)
        agent.learn()
        state_old = state_new

    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    print(score)

env.close()