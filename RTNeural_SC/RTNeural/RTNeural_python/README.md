# python examples

The best way to run all of these examples is to make a virtual environment in this "python" root directory, and run all the examples from this directory

**From the terminal:**

1. make sure you are in the RTNeural_python directory. For SuperCollider, this folder should be inside the RTNeural folder which contains the RTNeural.sc file and the HelpSource directory
2. if you haven't already, create a python virtual environment in that folder: 
//make the virtual environment (you only need to do this once)
> python -m venv venv

//activated the virtual environment
> source venv/bin/activate

//install the dependencies:
> pip install keras
> pip install tensorflow
> pip install torch

etc

3. run the training (for example)
> python 4Osc_MLP_torch/train_4Osc_torch.py

this will save the Pytorch model as 4Osc_torch

4. transform the torch training to RTNeural/keras and save as a json file
> python 4Osc_MLP_torch/torch_to_RTNeural_4Osc.py 

the RTNeural json file will be saved in the 4Osc_MLP_torch directory