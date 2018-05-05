import cv2
from directkeys import PressKey, ReleaseKey
from findscreen import grab_screen
import numpy as np
import pyautogui
from sklearn.cluster import KMeans
import threading
import time

screen_x_coor = 2079
screen_y_coor = 160
SCREEN_BOX = (screen_x_coor, screen_y_coor, screen_x_coor+950, screen_y_coor+520)
hood_y_coor = 290
horizon_y = 220
side_y = 50
keytime = 0.1

STRAITDIR = 0x11
RIGHTDIR = 0x20
LEFTDIR = 0x1E

VERTICES = np.array([[4, horizon_y + side_y],
                    [220, horizon_y], [580, horizon_y],
                    [800, horizon_y + side_y], [800, hood_y_coor], [4, hood_y_coor]])

for i in range(3, 0, -1):
    time.sleep(.4)
    print(i)


def t_key(key_a, key_b, key_c):
    PressKey(key_a)
    ReleaseKey(key_b)
    ReleaseKey(key_c)
    time.sleep(keytime)

# we want to start a thread with straight direction key held and other keys disregarded
def straight():
    straight_thread = threading.Thread(target=t_key, args=(STRAITDIR, RIGHTDIR, LEFTDIR))
    straight_thread.start()


def right():
    right_thread = threading.Thread(target=t_key, args=(RIGHTDIR, STRAITDIR, LEFTDIR))
    right_thread.start()

def left():
    left_thread = threading.Thread(target=t_key, args=(LEFTDIR, STRAITDIR, RIGHTDIR))
    left_thread.start()

def slope_cal(line):
    try:
        y = line[1] - line[3]
        x = line[0] - line[2]
        slope = np.divide(y, x)
    except ZeroDivisionError:
        slope = 100000
    finally:
        return slope

def drive(signval=None):
    sign = np.sum(np.sign(signval))
    if (sign == -2):
        right()
    elif (sign == 2):
        left()
    else:
        straight()


def drawVisualLines(img, lines):
    try:
        m = []
        for coordinates in lines:
            m.append(slope_cal(coordinates))
            coordinates = np.array(coordinates, dtype='uint32')
            cv2.line(img, (coordinates[0], coordinates[1]),
                     (coordinates[2], coordinates[3]), [255, 255, 255], 4)
    except TypeError as e:
        print("error trying to draw lines: {}".format(e))
    else:
        pass
        #drive(m)

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def processing(orig_img):
    # we take original image and do edge detection
    processed_img = cv2.Canny(orig_img, threshold1=100, threshold2=300)

    

    # cropping out parts of the frame that we are not interested in
    processed_img = roi(processed_img, [VERTICES])
    '''
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, np.array([]), 120, 20)
    #nlines = np.array([l[0] for l in lines])
    #drawVisualLines(processed_img, nlines)


    try:
        # implement KMeans algorithm
        n_lines = np.array([l[0] for l in lines])
        k_means = KMeans(n_cluster=2, random_state=0).fit(n_lines)
        drawVisualLines(processed_img, k_means.cluster_centers_)
    except (ValueError, TypeError) as e:
        print("KMEANS ERROR: {}".format(e))
    '''
    return processed_img


def main():
    while True:
        ti = time.time()
        screen = grab_screen(region=SCREEN_BOX)
        cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        new_screen = processing(screen)
        cv2.imshow('window2', new_screen)

        print('{:.2f} FPS'.format(1 / (time.time() - ti)))
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()


