# Pytorch Wavetable Lookup MLP Model

This is kind of a silly example. There are better ways to make a variable oscillator. But it is a proof of concept that this works.

This is basically the same as the "4Osc_MLP_keras" example, except the training runs in Pytorch instead of keras. The model then needs to be converted to keras for inference in RTNeural.

### Running the training

As with most of these trainings, it is best to create a virtual environment in the 'RTNeural/python' directory and run the training from there.

**From the terminal:**

1. make sure you are in the RTNeural/python directory
2. see the RTNeural/python/Readme.md file for setting up a virtual environment in the root directory

3. run the training
> python 4Osc_MLP_torch/train_4Osc_torch.py

this will save the Pytorch model as 4Osc_torch

4. transform the torch training to RTNeural/keras and save as a json file
> python 4Osc_MLP_torch/torch_to_RTNeural_4Osc.py 
