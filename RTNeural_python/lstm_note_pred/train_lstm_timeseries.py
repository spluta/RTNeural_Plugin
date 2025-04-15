# LSTM with Variable Length Input Sequences to One Character Output
import numpy as np
import keras
from keras import layers
from keras.utils import pad_sequences
import json

import argparse
import os

parser = argparse.ArgumentParser(
    description='Train a Linear only neural net on data from a file.')

parser.add_argument('-f', '--file', type=str, default='data.json', help='Path to the JSON file with the NN data.')
parser.add_argument('-o', '--outfile', type=str, default=None, help='Path to the output JSON training in Pytorch format.')
parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbose level for training. 0 = silent, 1 = progress bar, 2 = one line per epoch.')

args = parser.parse_args()
print(args)

# Load the data from a JSON file
json_path = os.path.abspath(args.file)
with open(json_path, 'r') as json_file:
    data = json.load(json_file)

json_dir = os.path.dirname(json_path)
raw_filename = os.path.splitext(os.path.basename(json_path))[0]

# Extract the data from the JSON object
in_vals = data['in_vals']
out_vals = data['out_vals']
layers_data = data['layers_data']
epochs = data['epochs']

hidden_size = layers_data[0][0]

max_len = 0
for i in range(len(in_vals)):
    if len(in_vals[i]) > max_len:
        max_len = len(in_vals[i])

print("max_len", max_len)
print("hidden_size", hidden_size)


print("in_vals", in_vals)
print("out_vals", out_vals)

in_min = 0
in_max = 0
for i in range(len(in_vals)):
    for j in range(len(in_vals[i])):
        if in_vals[i][j] < in_min:
            in_min = in_vals[i][j]
        if in_vals[i][j] > in_max:
            in_max = in_vals[i][j]
print("in_min", in_min)
print("in_max", in_max)

# convert list of lists to array and pad sequences if needed
X = pad_sequences(in_vals, maxlen=max_len, dtype='float32')
# reshape X to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], max_len, 1))
# normalize
X = (X - in_min) / (in_max - in_min + 1)

print(X)
print("X shape", X.shape[1])

# one hot encode the output variable
y = keras.utils.to_categorical(out_vals)

print("y shape", y.shape[1])

# create and fit the model
batch_size = 1
model = keras.Sequential()
model.add(layers.LSTM(hidden_size, input_shape=(X.shape[1], 1)))
model.add(layers.Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, y, epochs=epochs, batch_size=1, verbose=args.verbose)

# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

if args.outfile is None:
    out_temp = raw_filename+"_RTNeural.json"
    out_path = json_dir+"/"+out_temp

    keras_path = raw_filename+".keras"
    keras_path = json_dir+"/"+keras_path
else:
    out_path = args.outfile
    keras_path = args.outfile
    keras_path = os.path.splitext(keras_path)[0]+".keras"

print("keras path", keras_path)

from model_utils import save_model
save_model(model, out_path)

model.save(keras_path)

