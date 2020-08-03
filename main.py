import os, time, cv2, random
import numpy as np
from win32api import GetSystemMetrics
from PIL import ImageGrab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyautogui as pg

ROOT = os.getcwd()
reset_button = (818, 305) #beginner

def get_res():
    res = (GetSystemMetrics(0), GetSystemMetrics(1))
    box = (170, res[1]-170, res[0]-170, 170)

    im = ImageGrab.grab(box)
    print(type(im))

def screenGrab():
    # get bounding box for minesweeper prev_board
    # beginner coordinates:
    box = (706, 347, 930, 971)
    # expert box = (686, 267, 1463, 759)
    '''res = (GetSystemMetrics(0), GetSystemMetrics(1))

    pad = 170

    # padded screenshot
    box = (pad*3, pad, res[0]-pad*3, res[1]-pad)'''

    im = ImageGrab.grab(box)
    im.save(f'{ROOT}/board.png', 'PNG')

def click_random():
    corners = [(717, 358), (717, 558), (917, 358), (917, 558)]
    move = random.choice(corners)
    corners.remove(move)
    pg.click(x=move[0], y=move[1], duration=2)

def game_reset():
    pg.click(x=reset_button[0], y=reset_button[1], duration=2)

def sweep(board):
    prev_board = []
    while not np.array_equal(board, prev_board):
        click_random()
        prev_board = board
        screenGrab()
        board = mpimg.imread('board.png')
    game_reset()

def main():

    pg.FAILSAFE = True
    screenGrab()
    board = mpimg.imread('board.png')
    sweep(board)

if __name__ == '__main__':
    main()
