import torch
from torch.utils import data
from sklearn.preprocessing import Normalizer
import numpy as np
import os
import pandas
import h5py
from IPython import embed


class Load_Features:
    def __init__(self, tr_or_val = None):
        self.path_features = './audio_features/'
        self.tr_or_val = tr_or_val
        self.norm = Normalizer()
    
    def get_data(self):
        if self.tr_or_val == 'tr':
            self.path_input = self.path_features +'train/tr.hdf5'
            
        if self.tr_or_val == 'val':
            self.path_input =  self.path_features+'val/val.hdf5'
      
        self.all_files = []
        self.group = []
        def func(name, obj):     
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)
            elif isinstance(obj, h5py.Group):
                self.group.append(name)
        self.hf = h5py.File(self.path_input, 'r')
        self.hf.visititems(func)
        self.hf.close()

    
    def __len__(self):
            'Denotes the total number of samples'
            return len(self.all_files)
    
    def get_item(self,index):
        hf = h5py.File(self.path_input, 'r')

        emb = np.array(hf[self.all_files[index]])
        
        emb = emb.reshape(1, -1)
        normed_embed = self.norm.fit_transform(emb)
        
        ground_tr = np.array(int(self.all_files[index].split('/')[0]))
        #print(self.all_files[index],ground_tr)
        # normed_embed_tensor = torch.from_numpy(normed_embed).float()
        # ground_tr_tensor=torch.from_numpy(ground_tr).long()         
        return normed_embed,ground_tr