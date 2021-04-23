import torch
import random
import numpy as np
from collections import deque
from cart_model import Linear_QNet, QTrainer

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(4, 256, 2).to(
            device)  # 4 value state input, 2 action as output 0/1
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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
        # TODO: rework epsilon
        self.epsilon = 80 - self.n_games
        final_move = [0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move.index(1)

    # additional helper methods


# def train():
#     plot_scores = []
#     plot_mean_scores = []
#     total_score = 0
#     record = 0
#     agent = Agent()
#     while True:  # the good old while true
#         if agent.n_games == DISPLAY_GAMES:
#             game.start_display()
#
#         # get old state
#         state_old = agent.get_state(game)
#
#         # get move prediction
#         final_move = agent.get_action(state_old)
#
#         # perform move and get new state
#         if agent.n_games <= DISPLAY_GAMES:
#             reward, done, score = game.play_step(final_move, False)
#         else:
#             reward, done, score = game.play_step(final_move)
#         state_new = agent.get_state(game)
#
#         # train short memory with the information we just got by playing A in state S and getting reward R and ending
#         # up in S' (new state)
#         agent.train_short_memory(state_old, final_move, reward, state_new, done)
#
#         # remember for after epoch learning
#         agent.remember(state_old, final_move, reward, state_new, done)
#
#         # after every epoch
#         if done:
#             # train long memory, plot result
#             game.reset()
#             agent.n_games += 1
#             agent.train_long_memory()
#
#             if score > record:
#                 record = score
#                 agent.model.save()
#
#             print('Game', agent.n_games, 'Score', score, 'Record:', record)
#
#             plot_scores.append(score)
#             total_score += score
#             mean_score = total_score / agent.n_games
#             plot_mean_scores.append(mean_score)
#             plot(plot_scores, plot_mean_scores, agent.n_games)


# if __name__ == '__main__':
#     train()
