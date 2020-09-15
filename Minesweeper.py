import os, random
import numpy as np
from collections import deque
from DQN import *
from my_tensorboard import *

import warnings
warnings.filterwarnings('ignore')

ROOT = '/content/drive/My Drive/Minesweeper_AI/' # on Google Colab
ROOT = os.getcwd()

MEM_SIZE = 100_000
MEM_SIZE_MIN = 200
BATCH_SIZE = 64
DISCOUNT = 0.99 #gamma
MODEL_NAME = '256x4'
UPDATE_TARGET_EVERY = 5

# Environment settings
EPISODES = 10_000

# Learning settings
learn_rate = 0.0001
LEARN_DECAY = 0.99975
LEARN_MIN = 0.0001

# Exploration settings
epsilon = 0.9
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05

AGGREGATE_STATS_EVERY = 10
MIN_REWARD = -9  # For model save

REWARDS = {'win':2, 'lose':-2, 'progress':1, 'guess':-1}

class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        self.n_progress = 0
        self.n_wins = 0

        # Deep Q-learning Parameters
        self.rewards = REWARDS
        self.discount = DISCOUNT
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.model = create_dqn(self.learn_rate, self.state_im.shape, self.ntiles)

        # target model - this is what we predict against every step
        self.target_model = create_dqn(self.learn_rate, self.state_im.shape, self.ntiles)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

        self.tensorboard = ModifiedTensorBoard(log_dir=f'{ROOT}/logs/{MODEL_NAME}_lr{self.learn_rate}')

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(self.grid[row,col])

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors=='B')

    def get_board(self):
        board = self.grid.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.grid[x,y] != 'B':
                    coords.append((x,y))

        for coord in coords:
            board[coord] = self.count_bombs(coord)

        return board

    def get_state_im(self, state):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''

        state_2d = [t['value'] for t in state]
        state_2d = np.reshape(state_2d, (self.nrows, self.ncols, 1))

        state_im = np.zeros((self.nrows, self.ncols, 1))
        state_im[state_2d=='U'] = -1
        state_im[state_2d=='0'] = 0

        num_tiles = ~np.logical_or(state_2d == "U", state_2d == "0")
        state_im[num_tiles] = state_2d[num_tiles].astype(int) / 8

        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self.get_state_im(state)

        return state, state_im

    def click(self, action_index):
        coords = self.state[action_index]['coord']
        value = self.board[coord]

        # make state equal to board at given coordinates
        self.state[action_index]['value'] = self.board[coords]

        # reveal all neighbors if value is 0
        if value == 0:
            self.reveal_neighbors(coord)

    def reveal_neighbors(self, coord, processed=[]):
        state_df = pd.DataFrame(self.state)
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows) and
                    ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row,col))

                    index = state_df.index[state_df['coord'] == (row,col)].tolist()[0]
                    #self.click(index)

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0:
                        self.reveal_neighbors((row, col), processed=processed)

    def reset(self):
        self.n_progress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()

    def first_move(self):
        # random first move, ensure it is not a bomb
        move = np.random.randint(self.ntiles)
        coord = self.state[move]['coord']

        while self.board[coord] == 'B':
            move = np.random.randint(self.ntiles)
            coord = self.state[move]['coord']

        self.click(move)
        self.state_im = self.get_state_im(self.state)

    def get_action(self, state):
        board = state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-1]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(state, (1, self.nrows, self.ncols, 1)))
            moves[board!=-1] = 0
            move = np.argmax(moves)

        return move

    def step(self, action_index):
        done = False
        coords = self.state[action_index]['coord']

        # get neighbors before action
        neighbors = self.get_neighbors(coords)

        self.click(action_index)

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            done = True

        elif -1 not in self.state_im: # if win
            reward = self.rewards['win']
            done = True
            self.n_wins += 1

        else: # if progress
            # update state image
            self.state_im = self.get_state_im(self.state)

            if all(t=='U' for t in neighbors): # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']

            else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated clicks

        return self.state_im, reward, done

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done, episode):
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
        #self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)
