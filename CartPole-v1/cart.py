import gym
from cart_agent import Agent

env = gym.make('CartPole-v1')
n_actions = env.action_space.n
agent = Agent(gamma=0.9, epsilon=1.0, n_actions=n_actions)
epochs = 500

for i_episode in range(epochs):
    score = 0
    done = False
    state_old = env.reset()
    while not done:  # iterating over every timestep (state)
        # env.render()
        action = agent.get_action(state_old)
        state_new, reward, done, info = env.step(action)
        score += reward
        # print(action, state_old, state_new, reward, done, info)

        # train short memory with the information we just got by playing A in state S and getting reward R and ending
        # up in S' (new state)

        # agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember for after epoch learning
        agent.remember(state_old, action, reward, state_new, done)
        agent.train_long_memory()

        state_old = state_new

    # print("Episode finished after {} timesteps".format(t + 1))

    print(score)
    agent.n_games += 1
    # agent.train_long_memory()

env.close()
