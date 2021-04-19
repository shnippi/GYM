import gym

env = gym.make('Acrobot-v1')
env.reset()

print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
env.render()
action = 0
observation, reward, done, info = env.step(action)
#
# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()