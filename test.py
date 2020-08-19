import os, time, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pyautogui as pg

from collections import deque

from DQN import *
from tensorboard import *

# CONSTANTS ====

# Directories
ROOT = os.getcwd()
IMGS = f'{ROOT}/pics'

# Training settings
MEM_SIZE = 50_000
MEM_SIZE_MIN = 1_000
BATCH_SIZE = 64
DISCOUNT = 0.99

# Learning settings
learn_rate = 0.01
LEARN_DECAY = 0.9995
LEARN_MIN = 0.0001

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05

# Environment settings
EPISODES = 20_000

CONFIDENCES = {
    'unsolved': 0.99,
    'zero': 0.99,
    'one': 0.95,
    'two': 0.95,
    'three': 0.88,
    'four': 0.95,
    'five': 0.95,
    'six': 0.95,
    'seven': 0.95,
    'eight': 0.95
}

TILES = {
    'U': 'unsolved',
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four'
}

TILES2 = {
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
}

REWARDS = {'win':10, 'lose':-10, 'progress':1, 'guess':-1}
# ====

class Minesweeper:
    def __init__(self):
        pg.click((10,100)) # click on current tab so 'F2' resets the game
        self.reset()

        # Minesweeper Parameters
        self.mode, self.loc, self.dims = self.get_loc()
        self.nrows, self.ncols = self.dims[0], self.dims[1]
        self.ntiles = self.dims[2]
        self.board = self.get_board(self.loc)
        self.state = self.get_state(self.board)

        # Deep Q-learning Parameters
        self.rewards = REWARDS
        self.discount = DISCOUNT
        self.epsilon = epsilon
        self.model = create_dqn(LEARN_RATE, self.state.shape, self.ntiles)

        # target model - this is what we predict against every step
        self.target_model = create_dqn(LEARN_RATE, self.state.shape, self.ntiles)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

    def reset(self):
        pg.press('f2')

    def get_loc(self):
        '''
        obtain mode, screen coordinates and dimensions for Minesweeper board
        '''

        modes = {'beginner':(9,9,81), 'intermediate':(16,16,256), 'expert':(16,30,480)}
        boards = {mode: pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes.keys()}

        assert boards != {'beginner':None, 'intermediate':None, 'expert':None},\
            'Minesweeper board not detected on screen'

        for mode in boards.keys():
            if boards[mode] != None:
                diff = mode
                loc = boards[mode]
                dims = modes[mode]

        return diff, loc, dims

    def get_tiles(self, tile, bbox):
        '''
        Gets all locations of a given tile.
        Different confidence values are needed to correctly detect different tiles with grayscale=True
        '''
        conf = CONFIDENCES[tile]
        tiles = list(pg.locateAllOnScreen(f'{IMGS}/{tile}.png', region=bbox, grayscale=True, confidence=conf))

        return tiles

    def get_board(self, bbox):
        '''
        Gets the state of the board as a dictionary of coordinates and values,
        ordered from left to right, top to bottom
        '''

        all_tiles = [[t, self.get_tiles(TILES[t], self.loc)] for t in TILES]

        # for speedup; look for higher tiles if n of lower tiles < total ----
        count=0
        for value, coords in all_tiles:
            count += len(coords)

        if count < self.ntiles:
            higher_tiles = [[t, self.get_tiles(TILES2[t], self.loc)] for t in TILES2]
            all_tiles += higher_tiles
        # ----

        tiles = []
        for value, coords in all_tiles:
            for coord in coords:
                tiles.append({'coord': (coord[0], coord[1]), 'value': value})

        tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))

        '''self.nrows = sum(1 for i in tiles if i['coord'][0]==tiles[0]['coord'][0])
        self.ncols = sum(1 for i in tiles if i['coord'][1]==tiles[0]['coord'][1])'''

        i=0
        for x in range(self.nrows):
            for y in range(self.ncols):
                tiles[i]['index'] = (y, x)
                i+=1

        return tiles

    def get_state(self, board):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''
        board_2d = [t['value'] for t in board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols, 1))

        state = np.zeros((self.nrows, self.ncols, 1))
        state[board_2d=='U'] = -1
        state[board_2d=='0'] = 0

        num_tiles = ~np.logical_or(board_2d == "U", board_2d == "0")
        state[num_tiles] = board_2d[num_tiles].astype(int) / 8

        return state

    def get_action(self, state):
        board = self.state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-1]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(self.state, (1, self.nrows, self.ncols, 1)))
            moves[board!=-1] = 0 # ensure already revealed tiles are not chosen
            move = np.argmax(moves)

        return move

    def get_neighbors(self, action_index):
        board_2d = [t['value'] for t in self.board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols))

        tile = self.board[action_index]['index']
        x,y = tile[0], tile[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if (-1 < x < self.nrows and
                    -1 < y < self.ncols and
                    (x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(board_2d[col,row])

        return neighbors

    def step(self, action_index):
        done = False

        # number of solved tiles prior to move (initialized at 0)
        self.n_solved = self.n_solved_

        # get neighbors before clicking
        neighbors = self.get_neighbors(action_index)

        pg.click(self.board[action_index]['coord']) #, duration=0.1

        if pg.locateOnScreen(f'{IMGS}/oof.png', region=self.loc) != None: # if lose
            reward = self.rewards['lose']
            done = True
            self.n_solved_ = 0

        elif pg.locateOnScreen(f'{IMGS}/gg.png', region=self.loc) != None: # if win
            reward = self.rewards['win']
            done = True
            self.n_solved_ = 0

        else: # if progress
            self.board = self.get_board(self.loc)
            self.state = self.get_state(self.board)

            if all(t=='U' for t in neighbors): # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']

            else:
                reward = self.rewards['progress']

        return self.state, reward, done

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
                       shuffle=False, verbose=0)
                       #, shuffle=False, callbacks=[self.tensorboard]\
                       #if terminal_state else None)

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
