# Automated-Guitar Amplifier Modelling

This is a hacked version of Alec Wright's Guitar Amp Modeling. It modifies the original to only use LSTM or GRU RNN models, to be able to load files of any sample rate or bit rate, and run on an Apple GPU or CUDA (untested).

## Using this repository
View the help by running 'python dist_model_recnet.py -h'

This will pull up this information:

  -h, --help            show this help message and exit
  --in_file IN_FILE, -in IN_FILE
                        Name of the input file
  --target_file TARGET_FILE, -tar TARGET_FILE
                        Name of the target file
  --out_file OUT_FILE, -out OUT_FILE
                        the name of the json output file with the trained model
  --epochs EPOCHS, -eps EPOCHS
                        Max number of training epochs to run
  --learn_rate LEARN_RATE, -lr LEARN_RATE
                        Initial learning rate
  --init_len INIT_LEN, -il INIT_LEN
                        Number of sequence samples to process before starting weight updates
  --up_fr UP_FR, -uf UP_FR
                        For recurrent models, number of samples to run in between updating network weights, i.e the default argument updates every 1000 samples
  --pre_filt PRE_FILT, -pf PRE_FILT
                        Pre-filtering of input data, options are None, high_pass. default is high_pass
  --input_size INPUT_SIZE, -is INPUT_SIZE
                        1 for mono input data, 2 for stereo, etc - default is 1
  --output_size OUTPUT_SIZE, -os OUTPUT_SIZE
                        1 for mono output data, 2 for stereo, etc - default is 1
  --num_blocks NUM_BLOCKS, -nb NUM_BLOCKS
                        Number of recurrent blocks, default is 1
  --hidden_size HIDDEN_SIZE, -hs HIDDEN_SIZE
                        Recurrent unit hidden state size. default is 40
  --unit_type UNIT_TYPE, -ut UNIT_TYPE
                        LSTM or GRU
  --skip_con SKIP_CON, -sc SKIP_CON
                        is there a skip connection for the input to the output, default is 1
  --use_gpu USE_GPU, -gpu USE_GPU
                        Use the GPU if it is available, default is 1

### Running a training

The only required info for the training is the --in_file or -in and the --target_file or -tar. The --out_file or -out is recommended.

-in is the clean, undistorted signal
-tar is the distorted signal

they need to have the same number of samples

-out is the path to the json training.

other recommended parameters are:
-hs the number of hidden layers
-ut GRU or LSTM
-sc the skip connection can help a training converge quicker when the output signal is a distortion of the input signal


### Python Environment

you need to make a python environment in this repository

This repository requires a python environment with the 'pytorch', 'scipy', 'tensorboard' and 'numpy' packages installed. 

run: pip install torch scipy tensorboard numpy

### Processing Audio

The 'proc_audio.py' script loads a neural network model and uses it to process some audio, then saving the processed audio. This is also a good way to check if your python environment is setup correctly. Running the script with no arguments:

python proc_audio.py

will use the default arguments, the script will load the 'model_best.json' file from the directory 'Results/ht1-ht11/' and use it to process the audio file 'Data/test/ht1-input.wav', then save the output audio as 'output.wav'
Different arguments can be used as follows

python proc_audio.py 'path/to/input_audio.wav' 'output_filename.wav' 'Results/path/to/model_best.json'

### Training Script

Run the training from within the RTNeural_python virtual environment.
From the command line (from the AGAM_reduced directory):

python dist_model_recnet.py -in <infile> -tar <target file> -out <training destination> -hs <# of hidden layers> -ut <RNN format - LSTM or GRU> 

# Converting a Automated-GuitarAmpModelling Pytorch model to RTNeural format

Alec Wright's Automated-GuitarAmpModelling (used in Proteus, NeuralPi, Aida-X etc) uses Pytorch to save LSTM and GRU models to a Pytorch format JSON file. 

Aida-X files are already in RTNeural format! How nice.

Proteus json files and json files made with dist_model_recnet.json are in pytorch format and need to be converted. This folder provides a two scripts for transforming LSTM and GRU trainings to the Keras-based JSON format that RTNeural can read. One for converting GRU files and one for converting LSTM files.

Instructions

Make sure to start the RTNeural_python virtual environment before running the script!

From the command line from within the AGAM_reduced directory:

python AGAM_GRU_to_RTNeural.py -f <trained pytorch format json file>
--or--
python AGAM_LSTM_to_RTNeural.py -f <trained pytorch format json file>

