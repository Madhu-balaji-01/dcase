import torch
import torch.nn as nn
import torch.optim as optim
from data_engin import Data_Engin
from models.model import VGG_M, DCASE_PAST, DCASE_PAST2
from fit_model import Fit_Model

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-sm', '--save_model_address',
                    default = './model_zoo/dcase_tests/',
                    help = 'Path to save models.')
parser.add_argument('-s', '--spectra',
                    default = 'Mel_Spectrum',
                    help = 'Type of spectrogram: [Mel_Spectrum, Spectrum]')
parser.add_argument('-me', '--method',
                    default = 'post',
                    help = 'Timing to merge channels: [pre, post]')
parser.add_argument('-mo', '--mono',
                    default = 'mean',
                    help = 'Method to merge channels: [mean, diff]')
parser.add_argument('-n', '--network',
                    default = 'vgg_m',
                    help = 'Network to be used: [vgg_m, dcase1, dcase2]')

args = parser.parse_args()

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

no_class = 10
epoch = 30
lr = 0.0001
lr_state = { 'lr' : lr,
            'learning_rate_decay_start' : 10,
            'learning_rate_decay_every' : 3,
            'learning_rate_decay_rate' : 0.9
            }

pre_train_model_PATH = None

save_model_address = args.save_model_address
spectra_type = args.spectra
method = args.method
mono= args.mono

if args.network == 'vgg_m':
  network = VGG_M(no_class=no_class)
elif args.network == 'dcase1':
  network = DCASE_PAST(no_class=no_class)
elif args.network == 'dcase2':
  network = DCASE_PAST2(no_class=no_class)

train = Data_Engin(method=method, mono=mono, 
                   address='./dataset/dcase/evaluation_setup/modify_train.csv', 
                   spectra_type=spectra_type,
                   device=device, 
                   batch_size=64,
                   fs=48000,
                   n_fft=2048)

valid = Data_Engin(method=method, mono=mono,
                   address='./dataset/dcase/evaluation_setup/modify_evaluate.csv',
                   spectra_type=spectra_type,
                   device=device,
                   batch_size=64,
                   fs=48000,
                   n_fft=2048)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  network = nn.DataParallel(network, device_ids=[0, 1])

network = network.to(device)

# weights = torch.tensor([1., 1.2, 1., 1.5, 1., 2.5, 1.5]).to(device)
# criteria = nn.CrossEntropyLoss(weight=weights)
criteria = nn.CrossEntropyLoss()

optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(network.parameters(), lr=lr)

fit_model_class = Fit_Model(network=network,
                            device=device ,
                            optimizer = optimizer,
                            criteria = criteria,
                            lr_state = lr_state,
                            save_model_address=save_model_address)

# fit_model_class.load_model_zoo('./model_zoo/vgg-M-2/Epoch12-ACCtensor(78.2552).pth',
#                                 load_optim=True,
#                                 epoch_0=False)

results = fit_model_class.train_model(no_epoch=epoch, train_data_engine=train,
                                valid_data_engine=valid, save_mode=True)

# result = fit_model_class.test_model(test_data_engine=valid)
# result: train_acc, train_loss, valid_acc, valid_loss


print(f'Finished training: \nmodel: {args.network}')
cfg = vars(train)
print('\n'.join((f'{item}: {cfg[item]}' for item in cfg if item not in ['data', 'tranform', 'batch_itr'])))