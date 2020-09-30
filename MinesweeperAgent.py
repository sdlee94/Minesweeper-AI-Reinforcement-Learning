import random
import numpy as np
from collections import deque
from DQN import *
# use my_tensorboard2.py if using tensorflow v2+, use my_tensorboard.py otherwise
from my_tensorboard2 import *

import warnings
warnings.filterwarnings('ignore')

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer

# Learning settings
BATCH_SIZE = 64
learn_rate = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.01 #gamma

# based on https://github.com/jakejhansen/minesweeper_solver
REWARDS = {'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3}

# Exploration settings
epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 256 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 5

MODEL_NAME = f'conv{CONV_UNITS}x4_dense{DENSE_UNITS}x2_y{DISCOUNT}_minlr{LEARN_MIN}'

class MinesweeperAgent(object):
    def __init__(self, width, height, n_mines):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0

        # Deep Q-learning Parameters
        self.rewards = REWARDS
        self.discount = DISCOUNT
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.model = create_dqn(
            self.learn_rate, self.state_im.shape, self.ntiles, CONV_UNITS, DENSE_UNITS)

        # target model - this is what we predict against every step
        self.target_model = create_dqn(
            self.learn_rate, self.state_im.shape, self.ntiles, CONV_UNITS, DENSE_UNITS)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f'logs\\{MODEL_NAME}', profile_batch=0)

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

        state_im = [t['value'] for t in state]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self.get_state_im(state)

        return state, state_im

    def color_state(self, value):
        if value == -1:
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'

    def draw_state(self):
        state = self.state_im * 8.0
        state_df = pd.DataFrame(state.reshape((self.nrows, self.ncols)), dtype=np.int8)

        display(state_df.style.applymap(self.color_state))

    def click(self, action_index):
        coord = self.state[action_index]['coord']
        value = self.board[coord]

        # ensure first move is not a bomb
        if (value == 'B') and (self.n_clicks == 0):
            grid = self.grid.reshape(1, self.ntiles)
            move = np.random.choice(np.nonzero(grid!='B')[1])
            coord = self.state[move]['coord']
            value = self.board[coord]
            self.state[move]['value'] = value
        else:
            # make state equal to board at given coordinates
            self.state[action_index]['value'] = value

        # reveal all neighbors if value is 0
        if value == 0.0:
            self.reveal_neighbors(coord, clicked_tiles=[])

        self.n_clicks += 1

    def reveal_neighbors(self, coord, clicked_tiles):
        processed = clicked_tiles
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

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0.0:
                        self.reveal_neighbors((row, col), clicked_tiles=processed)

    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()

    def get_action(self, state):
        board = state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-1]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(state, (1, self.nrows, self.ncols, 1)))
            moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value
            move = np.argmax(moves)

        return move

    def step(self, action_index):
        done = False
        coords = self.state[action_index]['coord']

        # get neighbors before action
        neighbors = self.get_neighbors(coords)

        self.click(action_index)

        # update state image
        self.state_im = self.get_state_im(self.state)

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            done = True
            progress = 'Lose'

        elif np.sum(new_state_im==-0.125) == self.n_mines: # if win
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1
            progress = 'Win! :D'

        else: # if progress
            if all(t==-0.125 for t in neighbors): # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']
                progress = 'guess'

            else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated clicks
                progress = 'yes'

        return self.state_im, reward, done, progress

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
        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)
