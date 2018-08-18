from __future__ import division
import cv2
import os
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from itertools import islice
import glob

LIMIT = None

DATA_FOLDER = 'F:\\projects\\FaceEncoder\\1'
files = glob.glob(DATA_FOLDER+'/*.*')

def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    return resized

def return_data():

    X = []
    features = []

    for f1 in files:
        X.append(f1)

    for i in range(len(X)):
        img = plt.imread(X[i])
        #features.append(preprocess(img))
        features.append(img)

    features = np.array(features).astype('float32')

    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)

return_data()