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

def find_board():
    # obtain coordinates for Minesweeper board
    modes = ['beginner', 'intermediate', 'expert']
    boards = [pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes]
    for x in boards:
        if x != None:
            board = x

    assert board != None, 'Minesweeper board not detected on screen'
    return board

def get_state(bbox):
    all_tiles = [[t, list(pg.locateAllOnScreen(TILES[t], region=bbox))] for t in TILES]

    tiles = []
    for value, coords in all_tiles:
        for coord in coords:
            tiles.append({'coord': coord, 'value': value})
    tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))

    height = sum(1 for i in tiles if i['coord'][0]==tiles[0]['coord'][0])
    width = sum(1 for i in tiles if i['coord'][1]==tiles[0]['coord'][1])

    current_board = {i: {'coord': tile['coord'], 'value': tile['value'], 'action': None} for i, tile in enumerate(tiles)}

x,y = tiles[0][0], tiles[0][1]

pg.click(x,y)

def reset():
    loc = pg.locateOnScreen(f'{IMGS}/oof.png')
    pg.click(loc)
