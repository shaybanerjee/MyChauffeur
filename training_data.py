from getKeysPressed import key_check
import os
import cv2
import numpy as np
from findscreen import grab_screen
import time


screen_x_coor = 2079
screen_y_coor = 160
SCREEN_BOX = (screen_x_coor, screen_y_coor, screen_x_coor+950, screen_y_coor+520)


def neural_key_output(keys):
    # [A, W, D]
    output = [0, 0, 0, 0]
    if ('A' in keys):
        output[0] = 1
    elif ('D' in keys):
        output[2] = 1
    else:
        output[1] = 1

    return output


fname = 'trainingData.npy'

if os.path.isfile(fname):
    print('Loading previous data to update.')
    training_data = list(np.load(fname))
else:
    print('File does not exists, creating new file. ')
    training_data = []


def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    ti = time.time()
    while True:
        screen = grab_screen(region=SCREEN_BOX)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
        screen = cv2.resize(screen, (80, 60))
        keys = key_check()
        output = neural_key_output(keys)
        training_data.append([screen, output])
        #print("FRAME {} seconds".format(time.time() - ti))
        ti = time.time()
        if (len(training_data) % 500 == 0):
            print(len(training_data))
            np.save(fname, training_data)




if __name__ == '__main__':
    main()