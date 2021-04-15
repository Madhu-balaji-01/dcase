import torch
import torch.nn as nn
import torch.optim as optim
from data_engin import Data_Engin
from models.vgg_m import VGG_M
from models.dcase_past import DCASE_PAST
from fit_model import Fit_Model

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
# save_model_address = './model_zoo/dcase_tests/'
save_model_address = './model_zoo/dcase_4/'
pre_train_model_PATH = None

train = Data_Engin(address='/home/audio_server1/intern/dataset/dcase/evaluation_setup/modify_train.csv', spectra_type='Mel_Spectrum',
                      device=device, batch_size=64)

valid = Data_Engin(address='/home/audio_server1/intern/dataset/dcase/evaluation_setup/modify_evaluate.csv', spectra_type='Mel_Spectrum',
                      device=device, batch_size=64)

# train = Data_Engin(address='/home/audio_server1/intern/dataset/dcase/evaluation_setup/modify_train.csv', spectra_type='Spectrum',
#                       device=device, batch_size=64)

# valid = Data_Engin(address='/home/audio_server1/intern/dataset/dcase/evaluation_setup/modify_evaluate.csv', spectra_type='Spectrum',
#                       device=device, batch_size=64)

# test = Data_Engin(address='./dataset/test.csv', spectra_type='Spectrum',
#                       device=device, batch_size=64)

network = VGG_M(no_class=no_class)
# network = DCASE_PAST(no_class=no_class)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  network = nn.DataParallel(network, device_ids=[0, 1])

network = network.to(device)

# weights = torch.tensor([1., 1.2, 1., 1.5, 1., 2.5, 1.5]).to(device)
# criteria = nn.CrossEntropyLoss(weight=weights)
criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)

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

# result: train_acc, train_loss, valid_acc, valid_loss #
