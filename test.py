import os
import pyautogui as pg
from PIL import Image

ROOT = os.getcwd()
IMGS = f'{ROOT}/pics'

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

class Minesweeper:
    def __init__(self, TILES):
        #self.reset()
        self.TILES = TILES
        self.loc = self.find_board()
        self.state = self.get_state(self.loc)
        self.nrows = sum(1 for i in self.state if i['coord'][0]==self.state[0]['coord'][0])
        self.ncols = sum(1 for i in self.state if i['coord'][1]==self.state[0]['coord'][1])
        self.ntiles = self.nrows*self.ncols
        self.scaled = self.scale_state(self.state)
        self.n_solved_ = 0

    '''def reset(self):
        # restart game
        try:
            loc = pg.locateOnScreen(f'{IMGS}/oof.png')
        except:
            loc = pg.locateOnScreen(f'{IMGS}/reset.png')
        pg.click(loc, duration=0.3)'''

    def find_board(self):
        # obtain coordinates for Minesweeper board
        modes = ['beginner', 'intermediate', 'expert']
        boards = [pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes]

        assert boards != [None, None, None], 'Minesweeper board not detected on screen'

        for x in boards:
            if x != None:
                board = x

        return board

    def get_state(self, bbox):
        all_tiles = [[t, list(pg.locateAllOnScreen(self.TILES[t], region=bbox))] for t in self.TILES]

        tiles = []
        for value, coords in all_tiles:
            for coord in coords:
                tiles.append({'coord': (coord[0], coord[1]), 'value': value})

        tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))

        return tiles #{i: {'coord': tile['coord'], 'value': tile['value'], 'action': None} for i, tile in enumerate(tiles)}

    def scale_state(self, state):
        state = [t['value'] for t in state]
        state = np.reshape(state, (self.nrows, self.ncols, 1))

        scaled = np.zeros((self.nrows, self.ncols, 1))
        scaled[state=='U'] = -1
        scaled[state=='0'] = 0

        num_tiles = ~np.logical_or(state == "U", state == "0")
        scaled[num_tiles] = state[num_tiles].astype(int) / 8

        return scaled

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

x,y = tiles[0][0], tiles[0][1]

pg.click(x,y)
