# Pytorch Control-Rate MLP Model

This example takes any number of inputs and transforms that to any number of outputs using a Sequence of Linear models

### Running the training

As with most of these trainings, it is best to create a virtual environment in the 'RTNeural/python' directory and run the training from there.

**From the terminal:**

1. make sure you are in the RTNeural/python directory
2. see the RTNeural/python/Readme.md file for setting up a virtual environment in the root directory

3. run the training
> python MLP_torch_control/mlp_control.py -f MLP_torch_control/data.json

this will save the Pytorch model as 'mlp_control'

4. transform the torch training to RTNeural/keras and save as a json file
> python MLP_torch_control/torch_to_RTNeural_MLP.py 
