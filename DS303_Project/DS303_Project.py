import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import time
X_array = np.array([[[[]]]])

y_array = []
subdirs = [x[0] for x in os.walk('.', topdown= True)]
subdirs = subdirs[1:]
print(subdirs)

for i in range(len(subdirs)):
    a = time.time()
    m = os.listdir(subdirs[i])
    print(i,len(m))
    #print(os.getcwd() +subdirs[i][1:]+ str('/')+ m[0])
    for j in range(len(m)):
        try:
            image = cv2.imread(os.getcwd()+subdirs[i][1:]+ str('/')+ m[j])
            #image = image * 255
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if (image.shape == (300,300, 3)):
                image = np.expand_dims(image, axis = 0)
                #print(image.shape)
                if (i == 0 and j == 0):
                    X_array = image
                    y_array.append(i)
                else:
                    X_array = np.concatenate((X_array, image), axis = 0)
                    y_array.append(i)
        except OSError:
            print(i, j)
    print(time.time() - a)
print(X_array.shape)

y_array = np.array(y_array)
print(y_array)
print(len(y_array))

print(len(set(y_array)))

print(set(y_array))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
X_array, y_array = shuffle(X_array, y_array)
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, shuffle = True)

print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)

K = len(set(y_train))

i = Input(shape = X_train[0].shape)
x = Conv2D(32, (3,3), strides = 2, padding = 'valid', activation = 'relu')(i)
x = MaxPool2D(pool_size= (3,3), strides = (1,1))(x)
x = Conv2D(64, (3,3), strides = 2, padding = 'valid', activation = 'relu')(x)
#x = MaxPool2D(pool_size= (3,3), strides = (1,1))(x)
x = Conv2D(128, (3,3), strides = 2, padding = 'valid', activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512,activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation = 'softmax')(x)

model = Model(i,x)
print(model.summary())
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
r = model.fit(X_train, y_train, validation_data= (X_test, y_test), epochs = 30, batch_size = 100)

model.save_weights("weights_folder/weights")

