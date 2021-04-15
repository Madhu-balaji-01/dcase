from au_transform import Audio_Transform
import pandas as pd
import random
import torch
import math

class Data_Engin:

    def __init__(self, address=None,
                 spectra_type=None, device=None, batch_size=64,
                 fs=16000, time=10, n_fft = 1024,
                 win_len=0.05, hop_len=0.02):

        self.data_address = address
        # self.data_type = type
        self.device = device
        self.batch_size = batch_size
        self.spectra_type = spectra_type
        self.para = {}

        self.para['fs'] = fs
        self.para['time'] = time
        self.para['n_fft'] = n_fft
        self.para['win_length'] = int(win_len * fs)
        self.para['hop_length'] = int(hop_len * fs)

        self.transform = Audio_Transform(spectra_type=self.spectra_type,
                                         device=self.device,
                                         para=self.para)

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
        # aud_list = []
        # label = []
        for i in range(len_data):
            data.append([data_raw[key[0]][i], int(data_raw[key[2]][i])])
            # aud_list.append(data_raw[key[0]][i])
            # label.append(data_raw[key[1]][i])
        return data

    def mini_batch(self):

        if self.batch_itr >= self.no_batches:
            self.batch_itr = 0
            # self.shuffle_data()
            # print('shuffle')

        start = int(self.batch_itr*self.batch_size)
        end = int((self.batch_itr+1)*self.batch_size)
        self.batch_itr += 1

        aud_list = [self.data[i][0] for i in range(start,end)]
        label = [self.data[i][1] for i in range(start,end)]

        x, y = self.transform.main(aud_list,label)
        return x, y


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test = Data_Engin(address='test.csv', spectra_type='Mel_Spectrum',
                      device=device, batch_size=2)

    x, y = test.mini_batch()
















