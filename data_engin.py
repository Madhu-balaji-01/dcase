import os
import pandas as pd
from au_transform import Audio_Transform
from load_features import Load_Features
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import random
import torch
import math
from torch.utils import data
import h5py
from IPython import embed

class Data_Engin:

    def __init__(self, method='post', mono='mean', address=None,
                 spectra_type=None, device=None, batch_size=16,
                 fs=48000, time=1, n_fft = 1024, n_mels=128,
                 win_len=1024, hop_len=512, alpha = 0, spec_aug = False, 
                manipulate= False, tr_or_val = None):
        self.method = method
        self.mono = mono
        self.data_address = address
        self.device = device
        self.batch_size = batch_size
        self.spectra_type = spectra_type
        self.para = {}
        
        # Mixup
        self.alpha = alpha
        
        # SpecAug
        self.spec_aug = spec_aug
        
        # Audio manipulation
        self.manipulate = manipulate
        
        self.para['fs'] = fs
        self.para['time'] = time
        self.para['n_fft'] = n_fft
        self.para['n_mels'] = n_mels
        self.para['win_length'] = win_len
        self.para['hop_length'] = hop_len

        self.transform = Audio_Transform(method=self.method,
                                         mono=self.mono,
                                         spectra_type=self.spectra_type,
                                         device=self.device,
                                         para=self.para,
                                         spec_aug=self.spec_aug,
                                         manipulate = self.manipulate)
        
                
        self.data = self.read_csv()
        self.no_batches = math.floor(len(self.data)/self.batch_size)
        self.batch_itr = 0
        
        # Feature extraction
        self.tr_or_val = tr_or_val
        self.features = Load_Features(self.tr_or_val)
        self.features.get_data()
        self.no_batches_features = math.floor(self.features.__len__()/64)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
       

    def shuffle_data(self):
        random.shuffle(self.data)

    def read_csv(self):
        data_raw = pd.read_csv(self.data_address)
        key = [key for key in data_raw.keys()]
        len_data = len(data_raw[key[0]])
        data = []

        for i in range(len_data):
            data.append([data_raw[key[0]][i], int(data_raw[key[2]][i])])

        return data

    def mixup(self, x,y,alpha):
        lam = np.random.beta(alpha, alpha)
        shuffle = torch.randperm(x.shape[0])
        mixed_x = lam * x + (1 - lam) * x[shuffle, :]
        y_a, y_b = y, y[shuffle]
        return mixed_x, y_a, y_b, lam

    def mini_batch(self):
        if self.batch_itr >= self.no_batches:
            self.batch_itr = 0

        start = int(self.batch_itr*self.batch_size)
        end = int((self.batch_itr+1)*self.batch_size)
        self.batch_itr += 1

        aud_list = [self.data[i][0] for i in range(start,end)]
        label = [self.data[i][1] for i in range(start,end)]

        x, y = self.transform.main(aud_list,label)
        
        # Mixup
        if (self.alpha != 0):
            inputs, targets_a, targets_b, lam = self.mixup(x, y,self.alpha)
            return inputs, y, targets_a, targets_b,lam
        
        else:
            return x,y    

    def mini_batch_features(self):
        if self.batch_itr >= self.no_batches:
            self.batch_itr = 0

        start = int(self.batch_itr*64)
        end = int((self.batch_itr+1)*64)
        self.batch_itr += 1
        
        batch_features = np.empty((0,512))
        batch_labels = np.empty(0)
        for i in range(start,end):
            batch_embed, batch_label = self.features.get_item(i)
            # batch_embed = Variable(batch_embed).contiguous()
            # batch_label = Variable(batch_label).contiguous()
            # batch_features.append(batch_embed)
            # batch_labels.append(batch_label)
            
            # batch_features = torch.cat((batch_features, batch_embed), 0)
            # batch_labels = torch.cat((batch_labels,batch_label),0)
            batch_features = np.append(batch_features, batch_embed, axis=0)
            batch_labels = np.append(batch_labels, [batch_label], axis=0)
        # batch_features = torch.cat(batch_features, dim=0)
        # batch_labels = torch.cat(batch_labels, dim=0)
        batch_features = torch.from_numpy(batch_features)
        batch_labels = torch.from_numpy(batch_labels)
        
        # batch_embed = batch_embed.to(self.device)
        # batch_label = batch_label.to(self.device) 
        
        return batch_features,batch_labels
        

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test = Data_Engin(address='test.csv', spectra_type='Mel_Spectrum',
                      device=device, batch_size=2)

    x, y = test.mini_batch()
















