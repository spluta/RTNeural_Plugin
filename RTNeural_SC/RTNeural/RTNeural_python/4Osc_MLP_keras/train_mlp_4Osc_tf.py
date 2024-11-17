import numpy as np

from keras.src.callbacks.learning_rate_scheduler import LearningRateScheduler
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()


import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_path)
from model_utils import save_model


# Create the mlp model and define the loss function and optimizer
mlp_model = keras.Sequential()
mlp_model.add(keras.layers.InputLayer(shape=(None, 2)))
mlp_model.add(keras.layers.Dense(16, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(32, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(64, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(128, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(64, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(32, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(16, activation="sigmoid"))
mlp_model.add(keras.layers.Dense(1, kernel_initializer="orthogonal", bias_initializer='random_normal'))

# Set the buffer size for the wavetable
buf_size = 65536
num_waves = 4

# fill the training data with the 4 simple waveforms
sin = np.sin(np.linspace(0, 1-(1./buf_size), buf_size)*2*np.pi)
tri = np.concatenate((np.linspace(0, 1, buf_size//4), np.linspace(1, -1, buf_size//2+1)[1:], np.linspace(-1, 0-(1/buf_size), buf_size//4+1)[1:]))
square = np.concatenate((np.linspace(0, 1, 2), np.linspace(1, 1, buf_size//2-2), np.linspace(1, -1, 2), np.linspace(-1, -1, buf_size//2-2)))
saw = np.concatenate((np.linspace(0, 1, 2), np.linspace(1, -1, buf_size-1)[1:]))

y_train = np.concatenate((sin, tri, square, saw))

# make the input data
# in this case the nn will get two inputs:
# the phase of a waveform, then a value between 0 and 1 to interpolate between the waveforms, 0 should be the first waveform, 1 the last

X_train = np.linspace(0, 1, buf_size)

X_train_a = np.vstack((X_train,np.zeros_like(X_train))).reshape((-1,),order='F')
X_train_b = np.vstack((X_train,np.zeros_like(X_train)+1/(num_waves-1))).reshape((-1,),order='F')
X_train_c = np.vstack((X_train,np.zeros_like(X_train)+2/(num_waves-1))).reshape((-1,),order='F')
X_train_l = np.vstack((X_train,np.ones_like(X_train))).reshape((-1,),order='F')


# concatenate the training data and move the data to the gpu
X_train = np.concatenate((X_train_a, X_train_b, X_train_c, X_train_l))

X_train = X_train.reshape(X_train.size//2, 1,2)

# Compile the model
mlp_model.compile(optimizer="adam", loss="mean_squared_error")

def step_decay(epoch):
    loss = str (int (mlp_model.evaluate(X_train, y_train)*1000000))
    # mlp_model.save("mlp_4osc_model_"+loss+".keras")
    if epoch < 3:
        lrate = 0.001
    elif epoch < 7:
        lrate = 0.0001
    elif epoch < 10:
        lrate = 0.00001
    else:
        lrate = 0.000001
    return lrate
lrate=LearningRateScheduler(step_decay)

# Train the mlp model
mlp_model.fit(X_train, y_train, epochs=20, callbacks=[lrate], batch_size=1)

loss = str (int (mlp_model.evaluate(X_train, y_train)*1000000))
mlp_model.save(dir_path+"/mlp_4osc_model.keras")

save_model(mlp_model, dir_path+"/mlp_4osc_keras.json")