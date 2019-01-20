# Import required libraries
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

# Read dataset and extract images and corresponding steering angles as features and labels
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
            
# Spilt the dataset into 80% training and 20% validation
samples = list(zip(images,angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)   

# Data preprocessing, Gaussian blur and convert to RGB
def preprocess(image):
    prepro_image = cv2.GaussianBlur(image, (3,3),0)
    prepro_image = cv2.cvtColor(prepro_image, cv2.COLOR_BGR2RGB)
    return prepro_image      
            
# Python generator to generate batches on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = batch_sample[0]
                angle = batch_sample[1]
                aug_img = preprocess(image)
                images.append(aug_img)
                angles.append(angle)
                images.append(cv2.flip(aug_img,1)) # Augment data by flipping images horizontally
                angles.append(angle*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)           
            
batch_size = 32
batch_per_epoch = 80
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples,batch_size = batch_size)     
            
            
            
# aug_images, aug_angles = [], []
# for image, angle in zip(images, angles):
#     aug_images.append(image)
#     aug_angles.append(angle)
#     aug_images.append(cv2.flip(image,1))
#     aug_angles.append(angle*-1.0)
        
# X_train = np.array(aug_images)        
# y_train = np.array(aug_angles)

# Model Architecture
model = Sequential()

# Normalization Layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) 

# Crop image to remove unwanted scene
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Layer 1: Convolution layer with 24 5x5 filters, (2,2) strides, RELU activation
model.add(Conv2D(24,(5,5), strides=(2,2)))
model.add(Activation('relu'))

# Layer 2: Convolution layer with 36 5x5 filters, (2,2) strides, RELU activation
model.add(Conv2D(36,(5,5), strides=(2,2)))
model.add(Activation('relu'))

# Layer 3: Convolution layer with 48 5x5 filters, (2,2) strides, RELU activation
model.add(Conv2D(48,(5,5), strides=(2,2)))
model.add(Activation('relu'))

# Layer 4: Convolution layer with 64 3x3 filters,  RELU activation
model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))

#Layer 5: Convolution layer with 64 3x3 filters,  RELU activation
model.add(Conv2D(64, kernel_size=(3,3)))
model.add(Activation('relu'))

# Flatten 
model.add(Flatten())
model.add(Dropout(0.3))

# Layer 6: Fully connected layer with 100 neurons, Dropout(0.3), RELU activation
model.add(Dense(100))
model.add(Dropout(0.3))

# Layer 7: Fully connected layer with 50 neurons, Dropout(0.3), RELU activation
model.add(Dense(50))
model.add(Dropout(0.3))

# Layer 8: Fully connected layer with 10 neurons, Dropout(0.3), RELU activation
model.add(Dense(10))
model.add(Dropout(0.3))

# Layer 9: Fully connected layer with 1 neurons
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer = Adam(lr=1e-3))
model.fit_generator(train_generator, steps_per_epoch= batch_per_epoch*batch_size, validation_data=validation_generator,  validation_steps=batch_size, epochs=2, verbose=1)
print('Saving...')
model.save('model.h5')
print('Saved')
        