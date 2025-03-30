# Converting a Automated-GuitarAmpModelling Pytorch model to RTNeural format

Alec Wright's Automated-GuitarAmpModelling (used in Proteus, NeuralPi, etc) uses Pytorch to save LSTM and GRU models to a Pytorch format JSON file. This folder provides a method for transforming LSTM and GRU trainings to the Keras-based JSON format that RTNeural can read.

### Running the training

As with most of these trainings, it is best to create a virtual environment in the 'RTNeural/python' directory and run the training from there.

**From the terminal:**

1. make sure you are in the RTNeural/python directory
2. see the RTNeural/python/Readme.md file for setting up a virtual environment in the root directory
3. install the dependencies
> pip install keras
> pip install numpy
> etc

Once you have this environment up and running with all dependencies installed, all python programs in the RTNeural/python folder should work.

4. transform the torch training to keras and save as a json file

###Converting AGAM LSTM Model to RTNeural format

> python Automated-GuitarAmpModelling/AGAM_LSTM_to_RTNeural.py -f "the file to convert.json"

to run the provided example:
> python Automated-GuitarAmpModelling/AGAM_LSTM_to_RTNeural.py -f "Automated-GuitarAmpModelling/JoyoExtremeMetal.json"

this will save the file "JoyoExtremeMetal_RTNeural.json" in the Automated-GuitarAmpModelling directory

this should also work with full paths to other trainings:
> Automated-GuitarAmpModelling/AGAM_LSTM_to_RTNeural.py -f '/Library/Audio/Presets/Proteus_Tone_Packs/PedalPack1/TS9_HighDrive.json'

this will save the file "TS9_HighDrive_RTNeural.json" in the Automated-GuitarAmpModelling directory

###Converting AGAM GRU Model to RTNeural format

> python Automated-GuitarAmpModelling/AGAM_GRU_to_RTNeural.py -f "the file to convert.json"