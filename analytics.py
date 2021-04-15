import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

from data_engin import Data_Engin
from models.vgg_m import VGG_M
from fit_model import Fit_Model

from tqdm import tqdm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def load_model(pre_trained_model_path, network):
    checkpoint = torch.load(pre_trained_model_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network

def infer(network, valid_data_engine):
    network.eval()
    data = valid_data_engine

    valid_loss = 0
    correct = 0
    total = 0
    data.shuffle_data()
    with torch.no_grad():
        for i in tqdm(range(data.no_batches)):  # range(valid.no_batch)
            # print('batch_processed', i)
            inputs, labels = data.mini_batch()
            labels = labels.long()
            outputs = network(inputs)

            _, predicted = torch.max(outputs.data, 1)
            # _, original = torch.max(labels.data, 0)
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

####################################################################################################

if __name__ == '__main__':

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation', 'happy', 'angry']
    classes = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park']
    
    no_classes = len(classes)

    test = Data_Engin(address='./dataset/dcase/evaluation_setup/modify_evaluate2.csv', spectra_type='Spectrum',
                          device=device, batch_size=64)

    network = VGG_M(no_class=no_classes)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      network = nn.DataParallel(network, device_ids=[0, 1])

    network = network.to(device)
    network = load_model('./model_zoo/dcase_vgg_m_sub_b64/Epoch27-ACCtensor(59.8772).pth', network)

    acc, all_targets, all_predicted = infer(network,test)

    # Compute confusion matrix
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=classes, normalize=True,
                          title= 'VGG-M'+' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
    # plt.savefig(os.path.join(path, opt.split + '_cm.png'))
    # plt.close()