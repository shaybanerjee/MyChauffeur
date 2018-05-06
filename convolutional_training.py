# this is our training model

import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCH=8
MODEL_NAME = 'mychauffeur-{}-epochs.model'.format(LR, 'mod_alexnet', EPOCH)

model = alexnet(WIDTH, HEIGHT, LR)

training_data = np.load('trainingData_v2.npy')

train = training_data[:-500]
test = training_data[-500:]

# feature set (screen captures)
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)

# targets / labels (direction movement)
Y = np.array([i[1] for i in train])

test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = np.array([i[1] for i in test])

# out-of-sample testing to see how well we are doing and some information for tensorboard
model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCH, validation_set=({'input': test_X}, {'targets': test_Y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# tensorboard --logdir=foo:C:/Users/shayo/PycharmProjects/selfDriving/log

model.save(MODEL_NAME)





