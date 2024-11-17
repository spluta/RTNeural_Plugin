import torch
import torch.nn as nn
import keras
from keras.layers import Input
from keras.models import Model
# from model_utils import save_model
import json
import argparse


parser = argparse.ArgumentParser(
    description='Convert the Pytorch training to RTNeural format.')

parser.add_argument('-f', '--file', type=str, default='MLP_control/mlp_training.pt', help='Path to the JSON file with the Pytorch training.')
parser.add_argument('-o', '--outfile', type=str, default=None, help='Path to the output JSON training in RTNeural format.')

args = parser.parse_args()

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_path)
from model_utils import save_model

pytorch_path = args.file

class MLP(nn.Module):
    def __init__(self, input_size, layers_data: list):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            # print(activation)
            if activation is not None:
                assert isinstance(activation, nn.Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
       
    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

model = torch.load(pytorch_path)

# print(model.layers)

layers = model.layers


# for layer in layers:
#     layer_str = str(layer)
#     print(layer_str)

# # # Create a Keras model with the same architecture
keras_input = Input(shape=(layers[0].in_features,))
keras_output = keras.layers.Dense(layers[0].out_features, activation='relu')(keras_input)

for i in range(2, len(layers), 2):
    # print(i, layers[i+1])
    match str(layers[i+1]):
        case 'ReLU()':
            activation = 'relu'
        case 'Sigmoid()':
            activation = 'sigmoid'
        case 'Tanh()':
            activation = 'tanh'
        case _:
            activation = 'linear'
    keras_output = keras.layers.Dense(layers[i].out_features, activation)(keras_output)

# keras_output = keras.layers.Dense(10, activation='sigmoid')(keras_output)
keras_model = Model(inputs=keras_input, outputs=keras_output)

# print(keras_model.summary())

# # # Copy the weights from PyTorch model to Keras model
with torch.no_grad():
    keras_model.layers[1].set_weights([model.layers[0].weight.T.numpy(), model.layers[0].bias.numpy()])
    keras_model.layers[2].set_weights([model.layers[2].weight.T.numpy(), model.layers[2].bias.numpy()])
    keras_model.layers[3].set_weights([model.layers[4].weight.T.numpy(), model.layers[4].bias.numpy()])
    keras_model.layers[4].set_weights([model.layers[6].weight.T.numpy(), model.layers[6].bias.numpy()])
    keras_model.layers[5].set_weights([model.layers[8].weight.T.numpy(), model.layers[8].bias.numpy()])

# Generate a sample input
sample_input = torch.randn(1, 2)

# Set PyTorch model to evaluation mode
model.eval()

# Get PyTorch model prediction
with torch.no_grad():
    pytorch_output = model(sample_input)
    print("pytorch: ", pytorch_output)

# Get Keras model prediction
keras_output = keras_model.predict(sample_input.numpy(), verbose = 0)

# Convert Keras output to a PyTorch tensor
keras_output_tensor = torch.tensor(keras_output)
print("keras: ", keras_output_tensor)

# Compare predictions
if torch.allclose(pytorch_output, keras_output_tensor, atol=1e-6):
    print("Verification: The PyTorch and Keras models produce the same output. Saving to file.")
    if args.outfile is not None:
        print("Saving to: ", args.outfile)
        save_model(keras_model, args.outfile)
    else:
        print("Saving to: ", dir_path+"/mlp_RTNeural.json")
        save_model(keras_model, dir_path+"/mlp_RTNeural.json")
else:
    print("NO Verification: The PyTorch and Keras models produce different outputs.")


