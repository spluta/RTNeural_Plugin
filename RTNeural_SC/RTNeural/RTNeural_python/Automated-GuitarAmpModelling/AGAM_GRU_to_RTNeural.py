import json
import torch
import torch.nn as nn
import keras
from keras.layers import Input
from keras.models import Model

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_path)
from model_utils import save_model
from rnn_utils import transpose
from rnn_utils import swap_rz

import argparse

parser = argparse.ArgumentParser(
    description='Convert a Automated-GuitarAmpModelling LSTM model from PyTorch to Keras for use with RTNeural.')
parser.add_argument('-f', '--file', type=str, required=True, help='Path to the Automated-GuitarAmpModelling format JSON file.')
parser.add_argument('-o', '--output', type=str, default=None, help='Path to the output JSON file in RTNeural format.')
args = parser.parse_args()
print(args)

with open(args.file) as file:
    data = json.load(file)

for key in data:
    print(key)

input_size = data['model_data']['input_size']
output_size = data['model_data']['output_size']
num_layers = data['model_data']['num_layers']
hidden_size = data['model_data']['hidden_size']
has_bias = data['model_data']['bias_fl']

for key in data['state_dict']:
    print(key)

# Create a PyTorch model instance
class TorchRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rec = torch.nn.GRU(input_size, hidden_size)
        self.lin = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, h0):

        x, h0 = self.rec(input, h0)
        return self.lin(x)+input[0][0][0], h0

# make the torch model and load the data
torch_model = TorchRNN()
for name, param in torch_model.named_parameters():
        print(name)
        param.data = torch.tensor(data['state_dict'][name])

# Create a Keras model with the same architecture
keras_model = keras.Sequential()
# batch_input_shape is necessary for stateful LSTM
keras_model.add(keras.layers.InputLayer(batch_input_shape=(32, 0, input_size)))
# has to be stateful with tanh activation and sigmoid recurrent activation
keras_model.add(keras.layers.GRU (hidden_size, stateful = True, activation="tanh", return_sequences=True, recurrent_activation="sigmoid", use_bias=has_bias, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal"))
keras_model.add(keras.layers.Dense(1, kernel_initializer="orthogonal", bias_initializer='random_normal'))

gru_weights_ih = data['state_dict']['rec.weight_ih_l0']
wvals_ih = transpose(gru_weights_ih)  
swap_rz(wvals_ih, hidden_size) 

gru_weights_hh = data['state_dict']['rec.weight_hh_l0']
wvals_hh = transpose(gru_weights_hh)
swap_rz(wvals_hh, hidden_size)

if has_bias:
    gru_bias_ih = data['state_dict']['rec.bias_ih_l0']
    gru_bias_hh = data['state_dict']['rec.bias_hh_l0']
    gru_bias = [gru_bias_ih, gru_bias_hh]
    swap_rz(gru_bias, hidden_size)
    
else:
    gru_bias_ih = [0]*hidden_size*3
    gru_bias_hh = [0]*hidden_size*3
    gru_bias = [gru_bias_ih, gru_bias_hh]

import numpy as np

# set the weights of the keras model lstm layer
keras_model.layers[0].set_weights([np.array(wvals_ih), np.array(wvals_hh), np.array(gru_bias)])

# set the weights of the keras model dense layer
keras_model.layers[1].set_weights([np.array(data['state_dict']['lin.weight']).reshape(hidden_size,1), np.array(data['state_dict']['lin.bias'])])

# in order for the Pytorch to be stateful, we need to keep track of the hidden state
h0 = torch.zeros(1, 1, hidden_size).requires_grad_()

# Create a list of shape (None, None, 1)

for i in range(10):
        input = torch.randn((input_size), dtype=torch.float32)

        torch_input = input.reshape(1, 1, input_size)

        a, h0 = torch_model.forward(torch_input, h0)
        print("Pytorch: ", a[0].detach().numpy())
        b = keras_model.predict(torch_input.numpy(), verbose=0)+np.array(input)[0]
        print("Keras: ", b[0])
        print("")

output_file= os.path.splitext(args.file)[0]+"_RTNeural.json"

if (args.output != None):
    output_file=args.output

save_model(keras_model, output_file)

print("Model saved to:", output_file)
