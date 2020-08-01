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

# obtain coordinates for Minesweeper board
modes = ['beginner', 'intermediate', 'expert']
boards = [pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes]
for x in boards:
    if x != None:
        board = x

unsolved_tile = Image.open(f'{IMGS}/unsolved.png')
tiles = pg.locateAllOnScreen(unsolved_tile)
tiles = list(tiles)

x,y = tiles[0][0], tiles[0][1]

pg.click(x,y)
