import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

training_data = np.load("trainingData.npy")

'''

for data in training_data:
    # we stored screen in first index
    img = data[0]
    # we stored directional choice in second index
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

'''

df = pd.DataFrame(training_data)
print(df.head())
print(Counter(df[1].apply(str)))

left_dir = []
right_dir = []
forward_dir = []

# removes linearity of data but every frame is unique for our CNN
shuffle(training_data)

for data in training_data:
    # we stored screen in first index
    img = data[0]
    # we stored directional choice in second index
    choice = data[1]

    if (choice == [1,0,0,0]):
        left_dir.append([img, choice])
    elif (choice == [0,1,0,0]):
        forward_dir.append([img, choice])
    elif (choice == [0,0,1,0]):
        right_dir.append([img, choice])
    else:
        print("no match?")


# balancing data
'''
forward_dir = forward_dir[:len(right_dir)][:len(left_dir)]
left_dir = left_dir[:len(forward_dir)]
right_dir = right_dir[:len(forward_dir)]
'''

balanced_data = forward_dir + left_dir + right_dir

shuffle(balanced_data)
print(len(balanced_data))
np.save('trainingData_v2.npy', balanced_data)





