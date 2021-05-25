from os import name
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from data_engin import Data_Engin
from models.model import ENSEMBLE, VGG_M, VGG_M2, DCASE_PAST, DCASE_PAST2
from fit_model import Fit_Model

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--save_model_address',
                    default = './model_zoo/dcase_tests/',
                    help = 'Path to save models.')
parser.add_argument('--spectra',
                    default = 'mel_spectrum',
                    help = 'Type of spectrogram: [mel_spectrum, spectrum]')
parser.add_argument('--method',
                    default = 'post',
                    help = 'Timing to merge channels: [pre, post]')
parser.add_argument('--mono',
                    default = 'mean',
                    help = 'Method to merge channels: [mean, diff]')
parser.add_argument('--epoch',
                    default = 30,
                    help = 'Number of epochs to run.')
parser.add_argument('--batch_size',
                    default = 16,
                    help = 'Batch size to be used.')
parser.add_argument('--n_mels',
                    default = 128,
                    help = 'Number of mel features to extract.')
parser.add_argument('--win_len',
                    default = 1024,
                    help = 'Window length to be used.')
parser.add_argument('--hop_len',
                    default = 102,
                    help = 'Hop length to be used.')
parser.add_argument('--alpha',
                    default= 0,
                    help= 'Alpha for mixup data augmentation. Set to zero if mixup is not desired.')

args = parser.parse_args()

class Main_Train:
  def __init__(self, attr):
    if not isinstance(attr, dict):
      sys.exit('InitializError: Please provide class attributes in dictionary.')
    
    for name, value in attr.items():
      self.__setattr__(name, value)
  
  def load_data_engin(self, train_addr, valid_addr):
    torch.cuda.empty_cache()
    self.train_addr = train_addr
    self.valid_addr = valid_addr

    self.train = Data_Engin(method=self.method,
                       mono=self.mono, 
                      address=self.train_addr,
                      spectra_type=self.spectra_type,
                      device=self.device, 
                      batch_size=self.batch_size,
                      fs=self.fs,
                      n_fft=self.n_fft,
                      n_mels=self.n_mels,
                      win_len=self.win_len,
                      hop_len=self.hop_len,
                      alpha = self.alpha)

    self.valid = Data_Engin(method=self.method,
                       mono=self.mono,
                      address=self.valid_addr,
                      spectra_type=self.spectra_type,
                      device=self.device, 
                      batch_size=self.batch_size,
                      fs=self.fs,
                      n_fft=self.n_fft,
                      n_mels=self.n_mels,
                      win_len=self.win_len,
                      hop_len=self.hop_len)

  def get_network(self, network_type, models, multiple_gpu=True):
    if network_type == 'single':
      network = next(iter(models.values()))
      
    elif network_type == 'ensemble':
      network = ENSEMBLE(model_a=models['model_a'],
                         model_b=models['model_b'],
                         no_class=self.no_class)

    if multiple_gpu:
      if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network, device_ids=[0, 1])
    
    return network.to(self.device)    

  def fit_and_train(self, network, optimizer, criteria, save_model_address=None, save_mode=True):
    if save_model_address:
      self.save_model_address = save_model_address
      
    fit_model_class = Fit_Model(network=network,
                              device=self.device,
                              optimizer=optimizer,
                              criteria=criteria,
                              lr_state=self.lr_state,
                              save_model_address=self.save_model_address,
                              alpha = self.alpha)

    fit_model_class.train_model(no_epoch=self.epoch, train_data_engine=self.train,
                                valid_data_engine=self.valid, save_mode=save_mode)
    
    return fit_model_class.network
  
  def show_model_config(self, network_name):
    cfg = vars(self.train)
    print(f'Finished training: \nmodel: {network_name}')
    print('\n'.join((f'{item}: {cfg[item]}' \
      for item in cfg if item not in ['data', 'transform', 'batch_itr'])))

if __name__ == '__main__':
  # hyper-parameters
  attr = {
    'save_model_address': args.save_model_address,
    'no_class': 10,
    'epoch': int(args.epoch),
    'lr': 0.0001,
    'lr_state': {'lr': 0.0001,
                'learning_rate_decay_start': 10,
                'learning_rate_decay_every': 3,
                'learning_rate_decay_rate': 0.9
                },
    'method': args.method,
    'mono': args.mono,
    'spectra_type': args.spectra,
    'batch_size': int(args.batch_size),
    'fs': 48000,
    'n_fft': int(args.win_len),
    'n_mels': int(args.n_mels),
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'win_len': int(args.win_len),
    'hop_len': int(args.hop_len),
    'alpha': float(args.alpha)
  }
  trained_models = dict()
  
  # main train class
  trainer = Main_Train(attr=attr)
  trainer.load_data_engin(train_addr='./dataset/dcase/evaluation_setup/modify_train.csv',
                             valid_addr='./dataset/dcase/evaluation_setup/modify_evaluate.csv')
  
  # --------------------------------------------------------------------------------------------------------- #
  # load first model
  model_a =  {'model_a': DCASE_PAST(no_class=trainer.no_class)}
  network = trainer.get_network('single', models=model_a, multiple_gpu=False)

  # train first model
  optimizer = optim.SGD(network.parameters(),
                        lr=trainer.lr,
                        momentum=0.9,
                        weight_decay=5e-4)
  criteria = nn.CrossEntropyLoss()
  trained_models['model_a'] = trainer.fit_and_train(network=network,
                                                    optimizer=optimizer,
                                                    criteria=criteria,
                                                    save_mode=False)
  
  trainer.show_model_config(trained_models['model_a'].__class__.__name__)
  
  # --------------------------------------------------------------------------------------------------------- #
  # load second model
  model_b =  {'model_b': DCASE_PAST2(no_class=trainer.no_class)}
  network = trainer.get_network('single', models=model_b, multiple_gpu=False)

  # train second model
  optimizer = optim.SGD(network.parameters(),
                        lr=trainer.lr,
                        momentum=0.9,
                        weight_decay=5e-4)
  criteria = nn.CrossEntropyLoss()
  trained_models['model_b'] = trainer.fit_and_train(network=network,
                                                    optimizer=optimizer,
                                                    criteria=criteria,
                                                    save_mode=False)
  
  trainer.show_model_config(trained_models['model_b'].__class__.__name__)
  
  # --------------------------------------------------------------------------------------------------------- #
  # freeze the models
  for model in trained_models.values():
    for param in model.parameters():
      param.requires_grad_(False)

  # load ensemble model with trained models
  network = trainer.get_network('ensemble', models=trained_models, multiple_gpu=True)
  
  ensemble_addr = '_'.join([
    trained_models['model_a'].__class__.__name__,
    trained_models['model_b'].__class__.__name__,
    ''
  ])
  trainer.save_model_address += ensemble_addr

  # train ensemble model
  optimizer = optim.SGD(network.parameters(),
                        lr=trainer.lr,
                        momentum=0.9,
                        weight_decay=5e-4)
  criteria = nn.CrossEntropyLoss()
  trained_models['ensemble_model'] = trainer.fit_and_train(network=network,
                                                    optimizer=optimizer,
                                                    criteria=criteria,
                                                    save_mode=True)
  
  trainer.show_model_config(trained_models['ensemble_model'].__class__.__name__)