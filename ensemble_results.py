from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn as nn
from scipy.stats.mstats import gmean

from data_engin import Data_Engin
from models.model import VGG_M, DCASE_PAST, DCASE_PAST2
from fit_model import Fit_Model

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--spectra',
                    default = 'Mel_Spectrum',
                    help = 'Type of spectrogram: [Mel_Spectrum, Spectrum]')

args = parser.parse_args()

def load_model(pre_trained_model_path, network_type, no_class):
    if network_type == 'vgg_m':
        network = VGG_M(no_class=no_class)
    elif network_type == 'dcase1':
        network = DCASE_PAST(no_class=no_class)
    elif network_type == 'dcase2':
        network = DCASE_PAST2(no_class=no_class)
    
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      network = nn.DataParallel(network, device_ids=[0, 1])
      
    network = network.to(device)

    checkpoint = torch.load(pre_trained_model_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    
    del checkpoint
    torch.cuda.empty_cache()
    
    return network

def infer(network, valid_data_engine):
    network.eval()
    data = valid_data_engine

    correct = 0
    total = 0
    data.shuffle_data()
    
    with torch.no_grad():
        for i in tqdm(range(data.no_batches)):
            inputs, labels = data.mini_batch()
            labels = labels.long()
            outputs = network(inputs)

            _, predicted = torch.max(outputs.data, 1)
            original = labels.data
            total += labels.size(0)
            correct += predicted.eq(original.data).cpu().sum()

            if i == 0:
                all_predicted = predicted
                all_targets = labels
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, labels), 0)

    acc = 100. * correct / total
    print("accuracy: %0.3f" % acc)

    return acc, all_targets, all_predicted

def custom_ensemble(arr, idx):
    
    def max_occ(arr_1d,idx):
        bcount = np.bincount(arr_1d)
        if bcount.max()>1:
            return np.argmax(bcount)
        else:
            return arr_1d[idx]
    
    return np.apply_along_axis(max_occ, axis=1, arr=arr.T, idx=int(idx))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    classes = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']
    no_class = len(classes)
    
    models = [
        {'idx' : 0,
         'network_address' : './model_zoo/dcase_4/Epoch23-ACCtensor(59.7656).pth',
         'method' : 'pre',
         'mono' : 'mean',
         'spectra_type' : 'Mel_Spectrum',
         'batch_size' : 64,
         'fs' : 16000,
         'n_fft' : 1024,
         'n_mels' : 500,
         'network' : 'vgg_m'
         },
        {'idx' : 1,
         'network_address' : './model_zoo/dcase_tests/1-ACC60.5748-vgg_m-pre-mean-64-mel500.pth',
         'method' : 'pre',
         'mono' : 'mean',
         'spectra_type' : 'Mel_Spectrum',
         'batch_size' : 64,
         'fs' : 16000,
         'n_fft' : 1024,
         'n_mels' : 500,
         'network' : 'vgg_m'
         },
        {'idx' : 2,
         'network_address' : './model_zoo/dcase_tests/2-ACC60.4353-vgg_m-post-mean-64-mel500.pth',
         'method' : 'post',
         'mono' : 'mean',
         'spectra_type' : 'Mel_Spectrum',
         'batch_size' : 64,
         'fs' : 16000,
         'n_fft' : 1024,
         'n_mels' : 500,
         'network' : 'vgg_m'
         }
        #  {'idx' : 3,
        #  'network_address' : './model_zoo/dcase_11/Epoch28-Acc63.739.pth',
        #  'method' : 'pre',
        #  'mono' : 'mean',
        #  'spectra_type' : 'Mel_Spectrum',
        #  'batch_size' : 64,
        #  'fs' : 16000,
        #  'n_fft' : 1024,
        #  'n_mels' : 500,
        #  'network' : 'vgg_m'
        #  }
    ]
    
    
    acc = dict()
    target = dict()
    prediction = dict()
    
    for model in models:
        torch.cuda.empty_cache()
        
        idx = str(model['idx'])
        print('idx:', idx)
        
        test = Data_Engin(method=model['method'], mono=model['mono'],
                          address='./dataset/dcase/evaluation_setup/modify_evaluate.csv',
                          spectra_type=model['spectra_type'],
                          device=device, batch_size=model['batch_size'],
                          fs=model['fs'], n_fft=model['n_fft'], n_mels=model['n_mels'])
    
        network = load_model(model['network_address'],
                            network_type=model['network'],
                            no_class=no_class)
    

        acc[idx], target[idx], prediction[idx] = infer(network=network, valid_data_engine=test)
    
    for item in prediction.values():
        print(item[:10])
    
    pred_arr = np.array([item.detach().cpu().numpy() for item in prediction.values()])

    ensemble_prediction = custom_ensemble(pred_arr.T, int(max(acc, key=acc.get)))

    # ensemble_prediction = gmean(pred_arr+1, axis=0)
    # ensemble_prediction = np.around(ensemble_prediction)
    # ensemble_prediction -= 1

    print(ensemble_prediction[:10])
        
    
    target_val = target['0'].detach().cpu().clone()
    ensemble_prediction = torch.as_tensor(ensemble_prediction).cpu()
    
    crt = ensemble_prediction.eq(target_val).cpu().sum()
    acc = 100. * crt / target_val.size(0)

    print(f'Model Accuracy:\n{acc.items()}')
    print('Final Accuracy:', float(acc))