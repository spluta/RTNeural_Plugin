# Pytorch Control-Rate MLP Model

This example takes any number of inputs and transforms that to any number of outputs using a MultiLayered Perceptron

### Running the training

As with most of these trainings, it is best to create a virtual environment in the 'RTNeural/python' directory and run the training from there.

**From the terminal:**

1. make sure you are in the RTNeural_python directory
2. see the RTNeural_python/Readme.md file for setting up a virtual environment in the root directory
    (these trainings need torch, keras, and numpy installed in the virtual environment)
    (to save onnx format files, install onnxscript (not required))

**Here is the most basic training using default datapoints**
**See the mlp_control_tutorial.scd file for a step by step training from inside SuperCollider**

3. run the training
> python MLP_control/mlp_control_train_convert.py -f MLP_control/trainings/dumb_data/data.json

this will save the Pytorch model as 'mlp_training' in pytorch (.pt), RTNeural (_RTNeural.json) and if onnxscript is installed ONNX (.onnx)

