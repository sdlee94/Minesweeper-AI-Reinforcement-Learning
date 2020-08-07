import os
import pyautogui as pg
from PIL import Image

ROOT = os.getcwd()
IMGS = f'{ROOT}/pics'

TILES = {
    'U': Image.open(f'{IMGS}/unsolved.png'),
    '0': Image.open(f'{IMGS}/zero.png'),
    '1': Image.open(f'{IMGS}/one.png'),
    '2': Image.open(f'{IMGS}/two.png'),
    '3': Image.open(f'{IMGS}/three.png'),
    '4': Image.open(f'{IMGS}/four.png'),
    '5': Image.open(f'{IMGS}/five.png'),
    '6': Image.open(f'{IMGS}/six.png'),
    '7': Image.open(f'{IMGS}/seven.png'),
    '8': Image.open(f'{IMGS}/eight.png'),
    'M': Image.open(f'{IMGS}/mine.png'),
}

class Minesweeper:
    def __init__(self, TILES):
        self.TILES = TILES
        self.loc = self.find_board()
        self.state = self.get_state(self.loc)

    def find_board(self):
        # get coordinates for Minesweeper board on screen
        modes = ['beginner', 'intermediate', 'expert']
        boards = [pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes]

        assert boards != [None, None, None], 'Minesweeper board not detected on screen'

        for x in boards:
            if x != None:
                board = x

        return board

    def get_state(self, bbox):
        # get current state of the board
        all_tiles = [[t, list(pg.locateAllOnScreen(self.TILES[t], region=bbox))] for t in self.TILES]

        tiles = []
        for value, coords in all_tiles:
            for coord in coords:
                tiles.append({'coord': (coord[0], coord[1]), 'value': value})
        tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))

        self.height = sum(1 for i in tiles if i['coord'][0]==tiles[0]['coord'][0])
        self.width = sum(1 for i in tiles if i['coord'][1]==tiles[0]['coord'][1])

        return tiles #{i: {'coord': tile['coord'], 'value': tile['value'], 'action': None} for i, tile in enumerate(tiles)}

    def scale_state(self):
        # get state of the board as numeric values
        state = [t['value'] for t in self.state]
        state = np.reshape(state, (self.nrow, self.ncol, 1))

        scaled = np.zeros((self.nrow, self.ncol, 1))
        scaled[state=='U'] = -1
        scaled[state=='0'] = 0

        num_tiles = ~np.logical_or(state == "U", state == "0")
        scaled[num_tiles] = state[num_tiles].astype(int) / 8

        return scaled

    def reset():
        # restart game
        loc = pg.locateOnScreen(f'{IMGS}/oof.png')
        pg.click(loc)

x,y = tiles[0][0], tiles[0][1]

pg.click(x,y)
