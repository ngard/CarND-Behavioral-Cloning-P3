import csv, cv2
import numpy as np
import os
from PIL import Image
import sklearn

import random

train_samples = []
validation_samples = []

def sample_generator(validation_rate = 0.2):
    record_path = "./record/"
    for record in os.listdir(record_path):
        with open(record_path + record + "/driving_log.csv") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                source_path = line[0]
                filename = source_path.split('/')[-1]
                image_path = record_path + record + "/IMG/" + filename
                #image = cv2.imread(current_path)
                measurement = float(line[3])
                if random.random() <= validation_rate:
                    validation_samples.append((image_path,measurement))
                else:
                    train_samples.append((image_path,measurement))

sample_generator()
                    
batch_size = 64
if batch_size%2 != 0:
    print("batch size should be even number.")
    quit()

def generator(samples):    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size//2):
            batch_samples = samples[offset:offset+batch_size//2]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                file_path = batch_sample[0]
                center_image = np.asarray(Image.open(file_path))
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)
                flipped_image = np.fliplr(center_image)
                flipped_angle = -1 * center_angle
                images.append(flipped_image)
                angles.append(flipped_angle)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

is_train = True

image_shape = (160,320,3)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Dropout, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=image_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

def simple(model):
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(1024))
    model.add(Dense(1))

def VGG16(model):
    # Block 1    160x320
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2     80x160
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3     40x 80 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4     20x 40
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Classification block 102400
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.7))
    model.add(Dense(1024, activation='relu', name='fc3'))
    model.add(Dropout(0.7))
    model.add(Dense(256, activation='relu', name='fc4'))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu', name='fc5'))
    model.add(Dropout(0.7))
    model.add(Dense(16, activation='relu', name='fc6'))
    model.add(Dropout(0.7))
    model.add(Dense(1))

def NVIDIA(model):
    model.add(Conv2D(24,(5,5),subsample=(2,2),activation='relu'))
    model.add(Conv2D(36,(5,5),subsample=(2,2),activation='relu'))
    model.add(Conv2D(48,(5,5),subsample=(2,2),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dropout(0.7))
    model.add(Dense(50))
    #model.add(Dropout(0.7))
    model.add(Dense(10))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    
#simple(model)
#VGG16(model)
NVIDIA(model)

# following code does not work due to a bug in packages.
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

if is_train:
    model.compile(optimizer='adam',loss='mse')
    #model.fit(X_train, y_train, batch_size=16, epochs=3, validation_split=0.2, shuffle=True)

    model.fit_generator(train_generator, (len(train_samples)*2-1)//batch_size+1,
                        validation_data=validation_generator,
                        validation_steps=(len(validation_samples)*2-1)//batch_size+1,
                        epochs=3)
    
    model.save('model.h5')
