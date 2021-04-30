import torch
import torch.nn as nn
import torch.optim as optim
from data_engin import Data_Engin
from models.model import VGG_M, DCASE_PAST, DCASE_PAST2, ENSEMBLE
from fit_model import Fit_Model

# setting hyper-parameters
torch.cuda.empty_cache()
save_model_address = './model_zoo/dcase_ensemble_2/'
batch_size = 16
no_class = 10
epoch = 30
lr = 0.0001
lr_state = { 'lr' : lr,
            'learning_rate_decay_start' : 10,
            'learning_rate_decay_every' : 3,
            'learning_rate_decay_rate' : 0.9
            }
method = 'post'
mono = 'mean'
spectra_type = 'Mel_Spectrum'

models = {
  'model_a': VGG_M(no_class=no_class),
  'model_b': DCASE_PAST2(no_class=no_class)
}
trained_models = dict()

# train individual models
for model_name, model_network in models.items():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  train = Data_Engin(method=method, mono=mono, 
                    address='./dataset/dcase/evaluation_setup/modify_train.csv', 
                    spectra_type=spectra_type,
                    device=device, 
                    batch_size=batch_size,
                    fs=16000,
                    n_fft=1024,
                    n_mels=500)

  valid = Data_Engin(method=method, mono=mono,
                    address='./dataset/dcase/evaluation_setup/modify_evaluate.csv',
                    spectra_type=spectra_type,
                    device=device,
                    batch_size=batch_size,
                    fs=16000,
                    n_fft=1024,
                    n_mels=500)

  network = model_network
  network = network.to(device)

  criteria = nn.CrossEntropyLoss()
  optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

  fit_model_class = Fit_Model(network=network,
                              device=device ,
                              optimizer = optimizer,
                              criteria = criteria,
                              lr_state = lr_state,
                              save_model_address=save_model_address)

  fit_model_class.train_model(no_epoch=epoch, train_data_engine=train,
                              valid_data_engine=valid, save_mode=False)
  
  trained_models[model_name] = fit_model_class.network

  cfg = vars(train)
  print(f'Finished training: \nmodel: {trained_models[model_name].__class__.__name__}')
  print('\n'.join((f'{item}: {cfg[item]}' \
    for item in cfg if item not in ['data', 'transform', 'batch_itr'])))

  del train, valid, device, network
  torch.cuda.empty_cache()

# -------------------------------------------------
# Ensemble Model

# freezing the models
for model in trained_models.values():
  for param in model.parameters():
    param.requires_grad_(False)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = Data_Engin(method=method, mono=mono, 
                   address='./dataset/dcase/evaluation_setup/modify_train.csv', 
                   spectra_type=spectra_type,
                   device=device, 
                   batch_size=16,
                   fs=16000,
                   n_fft=1024,
                   n_mels=500)

valid = Data_Engin(method=method, mono=mono,
                   address='./dataset/dcase/evaluation_setup/modify_evaluate.csv',
                   spectra_type=spectra_type,
                   device=device,
                   batch_size=16,
                   fs=16000,
                   n_fft=1024,
                   n_mels=500)

network = ENSEMBLE(model_a=trained_models['model_a'], model_b=trained_models['model_b'],
                   no_class=no_class)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  network = nn.DataParallel(network, device_ids=[0, 1])

network = network.to(device)

criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

ensemble_addr = '_'.join([
  network.module.model_a.__class__.__name__,
  network.module.model_b.__class__.__name__,
  ''
])

save_model_address += ensemble_addr

fit_model_class = Fit_Model(network=network,
                            device=device ,
                            optimizer = optimizer,
                            criteria = criteria,
                            lr_state = lr_state,
                            save_model_address=save_model_address)

ensemble = fit_model_class.train_model(no_epoch=epoch, train_data_engine=train,
                                valid_data_engine=valid, save_mode=True)

cfg = vars(train)
print(f'Finished training: \nmodel: {ensemble_addr}')
print('\n'.join((f'{item}: {cfg[item]}' \
  for item in cfg if item not in ['data', 'transform', 'batch_itr'])))