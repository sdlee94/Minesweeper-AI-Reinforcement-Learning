import os
import pyautogui as pg
from PIL import Image

# CONSTANTS ====

# Directories
ROOT = os.getcwd()
IMGS = f'{ROOT}/pics'

# Training settings
MEM_SIZE = 50_000
MEM_SIZE_MIN = 1_000
LEARN_RATE = 0.001
BATCH_SIZE = 64
DISCOUNT = 0.99

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Environment settings
EPISODES = 20_000

TILES = {
    'U': f'{IMGS}/unsolved.png',
    '0': f'{IMGS}/zero.png',
    '1': f'{IMGS}/one.png',
    '2': f'{IMGS}/two.png',
    '3': f'{IMGS}/three.png',
    '4': f'{IMGS}/four.png',
    '5': f'{IMGS}/five.png',
    '6': f'{IMGS}/six.png',
    '7': f'{IMGS}/seven.png',
    '8': f'{IMGS}/eight.png',
    'M': f'{IMGS}/mine.png',
}
# ====

class Minesweeper:
    def __init__(self):
        #self.reset()
        self.TILES = TILES
        self.loc = self.find_board()
        self.state = self.get_state(self.loc)
        self.nrows = sum(1 for i in self.state if i['coord'][0]==self.state[0]['coord'][0])
        self.ncols = sum(1 for i in self.state if i['coord'][1]==self.state[0]['coord'][1])
        self.ntiles = self.nrows*self.ncols
        self.scaled = self.scale_state(self.state)
        self.n_solved_ = 0

        #self.memory = ReplayBuffer(MEM_SIZE, self.scaled.shape, self.ntiles)

        self.model = create_dqn(LEARN_RATE, self.scaled.shape, self.ntiles)

        # target model - this is what we predict against every step
        self.target_model = self.create_dqn(LEARN_RATE, self.scaled.shape, self.ntiles)
        self.target_model.set_weights(self.model.get_weights())

    def find_board(self):
        # obtain coordinates for Minesweeper board
        modes = ['beginner', 'intermediate', 'expert']
        boards = [pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes]

        assert boards != [None, None, None], 'Minesweeper board not detected on screen'

        for x in boards:
            if x != None:
                board = x

        return board

    def get_board(self, bbox):
        '''
        Gets the state of the board as a dictionary of coordinates and values,
        ordered from left to right, top to bottom
        '''
        all_tiles = [[t, list(pg.locateAllOnScreen(self.TILES[t], region=bbox))] for t in self.TILES]

        tiles = []
        for value, coords in all_tiles:
            for coord in coords:
                tiles.append({'coord': (coord[0], coord[1]), 'value': value})

        tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))

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

    def step(self, action_index):

        # number of solved tiles prior to move (initialized at 0)
        self.n_solved = self.n_solved_

        pg.click(self.state[action_index]['coord'], duration=0.2)

        if pg.locateOnScreen(f'{IMGS}/oof.png', region=self.loc) != None: # if lose
            reward = -1
            done = True

        else:
            new_state = self.get_state(self.loc)
            new_state = self.scale_state(new_state)

            # update number of solved tiles
            self.n_solved_ = self.ntiles - np.sum(self.scaled == -1)

            if pg.locateOnScreen(f'{IMGS}/gg.png', region=self.loc) != None: # if win
                reward = 1
                done = True

            elif self.n_solved_ > self.n_solved: # if progress
                reward = 0.9
                done = False

            elif self.n_solved_ == self.n_solved: # if no progress
                reward = -0.3
                done = False

        print(reward, done, end='\r')
        return done
        #return new_state, reward, done

    def remember(self, state, action, reward, new_state, done)
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_move(self, state):
        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.randint(self.ntiles)
        else:
            moves = self.model.predict(state)
            move = np.argmax(moves)

        return move

    def reset(self):
        # restart game
        try:
            loc = pg.locateOnScreen(f'{IMGS}/oof.png')
        except:
            loc = pg.locateOnScreen(f'{IMGS}/reset.png')
        pg.click(loc, duration=0.3)
