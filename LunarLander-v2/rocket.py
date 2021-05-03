import gym
import numpy as np
from plot import simple_plot
from rocket_agent import Agent

env = gym.make('LunarLander-v2')
input_dims = len(env.reset())  # how many elements does the state representation have?
n_actions = env.action_space.n
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=4, n_actions=n_actions, eps_end=0.01, input_dims=[input_dims],
              lr=0.003)
scores, avg_scores, eps_history = [], [], []
epochs = 500

for epoch in range(epochs):
    score = 0
    done = False
    state_old = env.reset()
    # print(state_old[0].type)
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
    avg_scores.append(avg_score)

    print("epoch: ", epoch, "score: %.2f " % score, "avg_score: %.2f " % avg_score, "epsilon: %.2f" % agent.epsilon)
    simple_plot(scores, avg_scores, epoch)

env.close()
