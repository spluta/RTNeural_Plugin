import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import torch
import torch.optim as optim
import argparse
import time
import os
from scipy.io.wavfile import write


prsr = argparse.ArgumentParser(
    description='''This script implements training for neural network amplifier/distortion effects modelling. ''')

prsr.add_argument('--in_file', '-in', default='Data/input.wav', help='Name of the input file')
prsr.add_argument('--target_file', '-tar', default='Data/target.wav', help='Name of the target file')

prsr.add_argument('--out_file', '-out', default='model.json', help='the name of the json output file with the trained model')

# number of epochs and validation
prsr.add_argument('--epochs', '-eps', type=int, default=500, help='Max number of training epochs to run')
# TO DO


prsr.add_argument('--learn_rate', '-lr', type=float, default=0.005, help='Initial learning rate')
prsr.add_argument('--init_len', '-il', type=int, default=200,
                  help='Number of sequence samples to process before starting weight updates')
prsr.add_argument('--up_fr', '-uf', type=int, default=1000,
                  help='For recurrent models, number of samples to run in between updating network weights, i.e the '
                       'default argument updates every 1000 samples')

prsr.add_argument('--pre_filt', '-pf', default='high_pass', help='Pre-filtering of input data, options are None, high_pass. default is high_pass')


# arguments for the network structure
prsr.add_argument('--input_size', '-is', default=1, type=int, help='1 for mono input data, 2 for stereo, etc - default is 1')
prsr.add_argument('--output_size', '-os', default=1, type=int, help='1 for mono output data, 2 for stereo, etc - default is 1')
prsr.add_argument('--num_blocks', '-nb', default=1, type=int, help='Number of recurrent blocks, default is 1')
prsr.add_argument('--hidden_size', '-hs', default=40, type=int, help='Recurrent unit hidden state size. default is 40')
prsr.add_argument('--unit_type', '-ut', default='LSTM', help='LSTM or GRU')
prsr.add_argument('--skip_con', '-sc', default=1, help='is there a skip connection for the input to the output, default is 1')
prsr.add_argument('--use_gpu', '-gpu', default=1, type=int, help='Use the GPU if it is available, default is 1')

args = prsr.parse_args()

print(torch)



if args.use_gpu==1:
    if torch.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
else:
    device = torch.device('cpu')

print('device: ' + str(device))

if __name__ == "__main__":
    """The main method creates the recurrent network, trains it and carries out validation/testing """
    start_time = time.time()

    # model_name = args.model + args.device + '_' + args.unit_type + '_hs' + str(args.hidden_size) + '_pre_' + args.pre_filt

    if args.pre_filt == 'high_pass':
        args.pre_filt = [-0.85, 1]
    elif args.pre_filt == 'None':
        args.pre_filt = None

    # Generate name of directory where results will be saved
    # save_path = 'Results/'

    # Check if an existing saved model exists, and load it, otherwise creates a new model
    network = networks.SimpleRNN(input_size=args.input_size, unit_type=args.unit_type, hidden_size=args.hidden_size,
                                    output_size=args.output_size, skip=args.skip_con)
    network.save_state = False
    # network.save_model('model', save_path)

    network = network.to(device)

    # Set up training optimiser + scheduler + loss fcns and training info tracker
    optimiser = torch.optim.Adam(network.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)
    loss_functions = training.LossWrapper({'ESRPre': 0.75, 'DC': 0.25}, args.pre_filt, device)
    train_track = training.TrainTrack()
    # writer = SummaryWriter(os.path.join('runs2', model_name))

    # Load dataset
    dataset = dataset.DataSet()

    dataset.create_subset('train', frame_len=22050)
    dataset.load_file(args.in_file, args.target_file, 'train', device)
    print('Data loaded')
    # print(dataset.subsets['train'].data['target'][0].shape)

    # If training is restarting, this will ensure the previously elapsed training time is added to the total
    init_time = time.time() - start_time + train_track['total_time']*3600
    # Set network save_state flag to true, so when the save_model method is called the network weights are saved
    network.save_state = True
    patience_counter = 0

    # This is where training happens
    # the network records the last epoch number, so if training is restarted it will start at the correct epoch number

    best_loss = 1.0

    for epoch in range(train_track['current_epoch'] + 1, args.epochs + 1):
        ep_st_time = time.time()

        # print(f"Network is on device: {network.device}")
        # Run 1 epoch of training,
        epoch_loss = network.train_epoch(dataset.subsets['train'].data['in'][0],
                                         dataset.subsets['train'].data['out'][0],
                                         loss_functions, optimiser, 50, args.init_len, args.up_fr)

        out_file= os.path.splitext(args.out_file)[0]

        if not os.path.isabs(out_file):
            out_file = os.path.join(os.getcwd(), out_file)

        if epoch_loss.item() < best_loss:
            best_loss = epoch_loss.item()
            network.save_model(out_file+'_best.json') 

        print('epoch: ' + str(epoch) + ' loss: ' + str(epoch_loss.item()))
        print('current learning rate: ' + str(optimiser.param_groups[0]['lr']))
        scheduler.step(epoch_loss)
        val_ep_st_time = time.time()
        train_track.train_epoch_update(epoch_loss.item(), ep_st_time, time.time(), init_time, epoch)
        network.save_model(out_file+'.json')
