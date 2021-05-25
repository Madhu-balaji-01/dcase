from au_transform import Audio_Transform
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import torch
import math

class Data_Engin:

    def __init__(self, method='post', mono='mean', address=None,
                 spectra_type=None, device=None, batch_size=64,
                 fs=48000, time=1, n_fft = 1024, n_mels=128,
                 win_len=1024, hop_len=512, alpha = 0, spec_aug = False, manipulate= False):
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

        


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test = Data_Engin(address='test.csv', spectra_type='Mel_Spectrum',
                      device=device, batch_size=2)

    x, y = test.mini_batch()
















