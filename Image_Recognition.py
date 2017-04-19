import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

#inputting images????

#MODEL
model = sequential()
model.add(Convolution2D(32, 5, 5, strides=1, padding='same', use_bias='yes', bias_initializer='random_normal', input_shape=(28, 28, 1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(63, 5, 5, strides=1, padding='same', use_bias='yes', bias_initializer='random_normal'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(10))

#COMPILING prior to training
sgd = SGD(Lr=0.00001, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)