import os
import numpy as np
import pyautogui as pg

ROOT = os.getcwd()
IMGS = f'{ROOT}/pics'

EPSILON = 0.01

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

class MinesweeperAgentWeb(object):
    def __init__(self, model):
        pg.click((10,100)) # click on current tab so 'F2' resets the game
        self.reset()

        self.mode, self.loc, self.dims = self.get_loc()
        self.nrows, self.ncols = self.dims[0], self.dims[1]
        self.ntiles = self.dims[2]
        self.board = self.get_board(self.loc)
        self.state = self.get_state(self.board)

        self.epsilon = EPSILON
        self.model = model

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
        Different confidence values are needed to correctly find different tiles with grayscale=True
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

        # for speedup; look for higher tiles only if n of lower tiles < total ----
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

        state_im = [t['value'] for t in board]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def get_action(self, state):
        board = self.state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(self.state, (1, self.nrows, self.ncols, 1)))
            moves[board!=-0.125] = np.min(moves)
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
        #self.n_solved = self.n_solved_

        # get neighbors before clicking
        neighbors = self.get_neighbors(action_index)

        pg.click(self.board[action_index]['coord'])

        if pg.locateOnScreen(f'{IMGS}/oof.png', region=self.loc) != None: # if lose
            done = True

        elif pg.locateOnScreen(f'{IMGS}/gg.png', region=self.loc) != None: # if win
            done = True

        else: # if progress
            self.board = self.get_board(self.loc)
            self.state = self.get_state(self.board)

        return self.state, done
