# Reference: Box Of Hats (https://github.com/Box-Of-Hats)

import win32api as winapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if winapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys