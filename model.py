import csv, cv2
import numpy as np
import os

cv2.namedWindow('window')
        
images = []
measurements = []

def add_data(image,measurement):
    # cv2.imshow('window',image)
    # cv2.waitKey(1)
    images.append(image)
    measurements.append(measurement)

record_path = "./record/"

for record in os.listdir(record_path):
    with open(record_path + record + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = record_path + record + "/IMG/" + filename
            image = cv2.imread(current_path)
            measurement = float(line[3])
            add_data(image,measurement)
            image_flipped = np.fliplr(image)
            measurement_flipped = -measurement
            add_data(image_flipped,measurement_flipped)
    
cv2.destroyAllWindows()

is_train = False

image_shape = images[0].shape
    
X_train = np.array(images)
y_train = np.array(measurements)

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
    model.fit(X_train, y_train, batch_size=16, epochs=3, validation_split=0.2, shuffle=True)

    model.save('model.h5')
