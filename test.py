import os
import csv
import cv2
import numpy as np
import tensorflow
import pdb

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Activation
from keras.layers import Dropout, ELU, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

lines = []
dataPath = './data'
with open(dataPath + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)
        
images = []
angles = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename    = source_path.split('/')[-1]
        current_path= './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        angle = float(line[3]) 
        if i == 0:
            angles.append(angle)
        elif i == 1:
            angles.append(angle + 0.20)
        else:
            angles.append(angle - 0.20)

samples = list(zip(images,angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)           
def preprocess(image):
    prepro_image = cv2.GaussianBlur(image, (3,3),0)
    prepro_image = cv2.cvtColor(prepro_image, cv2.COLOR_BGR2RGB)
    return prepro_image      

idx = 1000
cv2.imwrite('img1.jpg',images[idx])
