import os, time
import cv2
from win32api import GetSystemMetrics
from PIL import ImageGrab
import pyautogui

ROOT = os.getcwd()
replay_button = (818, 305)

def get_res():
    res = (GetSystemMetrics(0), GetSystemMetrics(1))
    box = (170, res[1]-170, res[0]-170, 170)

    im = ImageGrab.grab(box)
    print(type(im))

def screenGrab():
    #beginner
    box = (692, 267, 945, 584)
    # expert box = (686, 267, 1463, 759)
    '''res = (GetSystemMetrics(0), GetSystemMetrics(1))

    pad = 170

    # padded screenshot
    box = (pad*3, pad, res[0]-pad*3, res[1]-pad)'''

    im = ImageGrab.grab(box)
    im.save(f'{ROOT}/full_screen.png', 'PNG')

def calibrate():
    img = cv2.imread(f'{ROOT}/full_screen.png')

def main():

    pyautogui.FAILSAFE = True
    screenGrab()
    #screenGrab()

if __name__ == '__main__':
    main()
