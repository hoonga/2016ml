import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, Adam
import shogi
import agents

""" modifications for applying the agent to the shogi module
are written
Anything else are copied from https://gist.github.com/EderSantana/c7222daa328f0e885093"""

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = 13*11
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, reward_t, state_tp1 = self.memory[idx][0]
            state_t, state_tp1 = state_t.vectorize().reshape((1,13*11)),\
                                 state_tp1.vectorize().reshape((1, 13*11))
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i] = reward_t + self.discount * Q_sa
        return inputs, targets

def next_turn(action):
    agent = agents.RandomAgent()
    if agent.make_move(1, action) is None:
        return (action, 0, True)
    else:
        move = agent.make_move(-1, action)
        if move is None:
            return (action, 1, True)
        else:
            return (move, 0, False)

if __name__ == "__main__":
    # parameters
    epsilon = 0.1  # exploration
    epoch = 1000
    max_memory = 5000
    hidden_size = 512
    batch_size = 50

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(13*11,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(1))
    model.compile(Adam(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("manmodel.h5")

    # Define environment/game
    r = shogi.Rule()

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        board = shogi.Board([[-2,-1,-3],[0,-4,0],[0,4,0],[3,1,2]],[])
        game_over = False
        # get initial input
        input_t = board

        while not game_over:
            # get next action
            input_tm1 = input_t
            if np.random.rand() <= epsilon:
                moves = r.gen_moves(1, board)
                if len(moves) == 0:
                    action = board
                else:
                    action = moves[np.random.randint(0, len(moves), size=1)]
            else:
                q = []
                moves = r.gen_moves(1, board)
                if len(moves) != 0:
                    for b in moves:
                        q.append(model.predict(b.vectorize().reshape((1,13*11))))
                    action = moves[q.index(max(q))]
                else:
                    action = board

            # apply action, get rewards and new state
            input_t, reward, game_over = next_turn(action)
            board = input_t
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            a = model.train_on_batch(inputs, targets)
            loss += a
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save("manmodel.h5", overwrite=True)