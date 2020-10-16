import os, sys

ROOT = os.getcwd()
sys.path.insert(1, f'{os.path.dirname(ROOT)}')

import warnings
warnings.filterwarnings('ignore')

from collections import deque
from minesweeper_env import *
# use my_tensorboard2.py if using tensorflow v2+, use my_tensorboard.py otherwise
from my_tensorboard2 import *
from DQN import *

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer

# Learning settings
BATCH_SIZE = 64
learn_rate = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma

# Exploration settings
epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 5

# Default model name
MODEL_NAME = f'conv{CONV_UNITS}x4_dense{DENSE_UNITS}x2_y{DISCOUNT}_minlr{LEARN_MIN}'

class DQNAgent(object):
    def __init__(self, env, model_name=MODEL_NAME, conv_units=64, dense_units=256):
        self.env = env

        # Deep Q-learning Parameters
        self.discount = DISCOUNT
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.model = create_dqn(
            self.learn_rate, self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)

        # target model - this is what we predict against every step
        self.target_model = create_dqn(
            self.learn_rate, self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f'logs\\{model_name}', profile_batch=0)

    def get_action(self, state):
        board = state.reshape(1, self.env.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1)))
            moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value
            move = np.argmax(moves)

        return move

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X,y = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE,
                       shuffle=False, verbose=0, callbacks=[self.tensorboard]\
                       if done else None)

        # updating to determine if we want to update target_model yet
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # decay learn_rate
        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)

if __name__ == "__main__":
    DQNAgent(MinesweeperEnv(9,9,10))
