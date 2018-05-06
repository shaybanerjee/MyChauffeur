import cv2
import time
from findscreen import grab_screen
from getKeysPressed import key_check
import os
from alexnet import alexnet
import numpy as np
from directkeys import PressKey, ReleaseKey
import threading
import random


screen_x_coor = 2079
screen_y_coor = 160
SCREEN_BOX = (screen_x_coor, screen_y_coor, screen_x_coor+950, screen_y_coor+520)
hood_y_coor = 290
horizon_y = 220
side_y = 50
keytime = 0.05

STRAITDIR = 0x11
RIGHTDIR = 0x20
LEFTDIR = 0x1E
BRAKE = 0x1F

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCH=8
MODEL_NAME = 'mychauffeur-{}-epochs.model'.format(LR, 'mod_alexnet', EPOCH)


def one_key(key_a, key_b, key_c):
    PressKey(key_a)
    ReleaseKey(key_b)
    ReleaseKey(key_c)
    time.sleep(keytime)

'''
def two_keys(key_a, key_b, key_c, key_d):
    PressKey(key_a)
    PressKey(key_b)
    ReleaseKey(key_c)
    ReleaseKey(key_d)
    time.sleep(keytime)

'''

def no_keys(key_a, key_b, key_c):
    ReleaseKey(key_a)
    ReleaseKey(key_b)
    ReleaseKey(key_c)
    time.sleep(keytime)



# we want to start a thread with straight direction key held and other keys disregarded
# threads are used because we would like to simulate a pause
# if we were to use one thread our FPS would drop significantly

def straight():
    if (random.random() <= 0.20):
        straight_thread = threading.Thread(target=one_key, args=(STRAITDIR, RIGHTDIR, LEFTDIR))
        straight_thread.start()
    else:
        idle_thread = threading.Thread(target=no_keys, args=(STRAITDIR, RIGHTDIR, LEFTDIR))
        idle_thread.start()


def right():
    right_thread = threading.Thread(target=one_key, args=(RIGHTDIR, STRAITDIR, LEFTDIR))
    right_thread.start()

def left():
    left_thread = threading.Thread(target=one_key, args=(LEFTDIR, STRAITDIR, RIGHTDIR))
    left_thread.start()

my_model = alexnet(WIDTH, HEIGHT, LR)
my_model.load(MODEL_NAME)

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    ti = time.time()
    stop=False
    while True:
        if not stop:
            screen = grab_screen(region=SCREEN_BOX)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
            screen = cv2.resize(screen, (80, 60))
            print('Frame in {} seconds'.format(time.time() - ti))
            to = time.time()
            # getting prediction on which direction to move based on screen
            predict = my_model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            # rounding elements in the array
            moves = list(np.around(predict))
            print(moves, predict)
            if (moves == [1,0,0,0]):
                # left
                left()
            elif (moves == [0,1,0,0]):
                # straight
                straight()
            elif (moves == [0,0,1,0]):
                # right
                right()
            else:
                print("unknown, won't move")
        keys = key_check()
        if 'T' in keys:
            if stop:
                stop = False
                time.sleep(1)
            else:
                stop = True
                ReleaseKey(STRAITDIR)
                ReleaseKey(RIGHTDIR)
                ReleaseKey(LEFTDIR)


if __name__ == '__main__':
    main()
