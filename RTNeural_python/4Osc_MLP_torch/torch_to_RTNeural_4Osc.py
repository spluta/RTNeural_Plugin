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

# first, define the Pytorch model

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressor, self).__init__()
        self._methods = []
        self._attributes = ["none"]
        self.fc1 = nn.Linear(input_size, hidden_size//2)  # Modified line
        self.fc2 = nn.Linear(hidden_size//2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size//2)
        self.fc5 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc6 = nn.Linear(hidden_size//4, hidden_size//8)
        self.fc7 = nn.Linear(hidden_size//8, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = self.fc7(x)
        return x

# Create a PyTorch model instance
model = MLPRegressor(2, 256, 1)

model_state_dict = torch.load(dir_path+"/4Osc_torch.pt")

model.load_state_dict(model_state_dict) # Load the model

print(model)

# Create a Keras model with the same architecture
keras_input = Input(shape=(2,))
keras_output = keras.layers.Dense(128, activation='sigmoid')(keras_input)
keras_output = keras.layers.Dense(256, activation='sigmoid')(keras_output)
keras_output = keras.layers.Dense(256, activation='sigmoid')(keras_output)
keras_output = keras.layers.Dense(128, activation='sigmoid')(keras_output)
keras_output = keras.layers.Dense(64, activation='sigmoid')(keras_output)
keras_output = keras.layers.Dense(32, activation='sigmoid')(keras_output)
keras_output = keras.layers.Dense(1)(keras_output)
keras_model = Model(inputs=keras_input, outputs=keras_output)

# # Copy the weights from PyTorch model to Keras model
with torch.no_grad():
    keras_model.layers[1].set_weights([model.fc1.weight.T.numpy(), model.fc1.bias.numpy()])
    keras_model.layers[2].set_weights([model.fc2.weight.T.numpy(), model.fc2.bias.numpy()])
    keras_model.layers[3].set_weights([model.fc3.weight.T.numpy(), model.fc3.bias.numpy()])
    keras_model.layers[4].set_weights([model.fc4.weight.T.numpy(), model.fc4.bias.numpy()])
    keras_model.layers[5].set_weights([model.fc5.weight.T.numpy(), model.fc5.bias.numpy()])
    keras_model.layers[6].set_weights([model.fc6.weight.T.numpy(), model.fc6.bias.numpy()])
    keras_model.layers[7].set_weights([model.fc7.weight.T.numpy(), model.fc7.bias.numpy()])

# Generate a sample input
sample_input = torch.randn(1, 2)

# Set PyTorch model to evaluation mode
model.eval()

# Get PyTorch model prediction
with torch.no_grad():
    pytorch_output = model(sample_input)

# Get Keras model prediction
keras_output = keras_model.predict(sample_input.numpy())

# Convert Keras output to a PyTorch tensor
keras_output_tensor = torch.tensor(keras_output)

# Compare predictions
if torch.allclose(pytorch_output, keras_output_tensor, atol=1e-6):
    print("Verification: The PyTorch and Keras models produce the same output. Saving to file.")
    save_model(keras_model, dir_path+"/4Osc_torch_RTNeural.json")
else:
    print("Verification: The PyTorch and Keras models produce different outputs.")


