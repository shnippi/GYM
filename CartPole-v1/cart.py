import gym
from cart_agent import Agent

env = gym.make('CartPole-v1')
env.reset()

agent = Agent()

for i_episode in range(10000):
    state_old = env.reset()
    for t in range(1000):  # number of timesteps
        env.render()
        action = agent.get_action(state_old)
        state_new, reward, done, info = env.step(action)
        # print(action, state_old, state_new, reward, done, info)

        # train short memory with the information we just got by playing A in state S and getting reward R and ending
        # up in S' (new state)
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember for after epoch learning
        agent.remember(state_old, action, reward, state_new, done)

        state_old = state_new

        if done:
            print("Episode finished after {} timesteps".format(t + 1))

            agent.n_games += 1
            agent.train_long_memory()

            break


env.close()
