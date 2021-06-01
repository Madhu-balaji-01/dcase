import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

class Fit_Model:

    def __init__(self, network, device,
                 lr_state, optimizer,
                 criteria, save_model_address=None, alpha = 0, is_baseline = False):

        self.network = network
        self.device = device
        self.optimizer = optimizer
        self.criteria = criteria
        self.lr_state = lr_state
        self.save_model_address = save_model_address

        self.alpha = alpha 
        self.pretrained_epoch = None
        self.pretrained_valid_acc = None
        self.is_baseline = is_baseline

    def __set_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def __clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-grad_clip, grad_clip)

    def lr_rate_setting(self, epoch):
        if epoch > self.lr_state['learning_rate_decay_start'] and self.lr_state['learning_rate_decay_start'] >= 0:
            frac = (epoch - self.lr_state['learning_rate_decay_start']) // self.lr_state['learning_rate_decay_every']
            decay_factor = self.lr_state['learning_rate_decay_rate'] ** frac
            self.current_lr = self.lr_state['lr'] * decay_factor
            self.__set_lr(self.optimizer, self.current_lr)  # set the decayed rate
        else:
            self.current_lr = self.lr_state['lr']
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        mixup_loss = lam * self.criteria(pred, y_a) + (1 - lam) * self.criteria(pred, y_b)
        return mixup_loss

    def save_model(self, epoch ,valid_acc, valid_loss):
        model_name_save = '{}Epoch{}-Acc{}.pth'.format(self.save_model_address,
                                                str(epoch),
                                                str(np.round(float(valid_acc),4)))
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': valid_loss,
                'acc':valid_acc,
                }, model_name_save)
        print('saved the model')
        self.best_acc = valid_acc

    def train_process(self, epoch, train_data_engine):
        self.network = self.network.train()
        self.lr_rate_setting(epoch)
        data = train_data_engine

        train_loss = 0
        correct = 0
        total = 0

        data.shuffle_data()
        print('no of batches', data.no_batches)
        for _ in tqdm(range(data.no_batches), dynamic_ncols=True):  # range(train.no_batch)
            inputs, labels = data.mini_batch()
            labels = labels.long()
            outputs = self.network(inputs)
            loss = self.criteria(outputs, labels)
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.__clip_gradient(self.optimizer, 0.1)
            self.optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            original = labels.data
            total += labels.size(0)
            correct += predicted.eq(original.data).cpu().sum()
            train_loss += loss.item()

        epoch_acc = 100. * correct/total
        epoch_loss = train_loss / data.no_batches

        print('Epoch:{} training loss:'
              ' {} training accuracy {}'.format(epoch + 1,
                                               np.round(epoch_loss,4),
                                               np.round(epoch_acc,4)))

        return epoch_acc, epoch_loss

    def train_process_mixup(self, epoch, train_data_engine):
        self.network = self.network.train()
        self.lr_rate_setting(epoch)
        data = train_data_engine

        train_loss = 0
        correct = 0
        total = 0

        data.shuffle_data()
        print('no of batched', data.no_batches)
        for _ in tqdm(range(data.no_batches), dynamic_ncols=True):  # range(train.no_batch)
            inputs, y, labels_a, labels_b, lam = data.mini_batch()
            inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
        
            labels_a = torch.tensor(labels_a, dtype=torch.long)
            labels_b = torch.tensor(labels_b, dtype=torch.long)
            
            outputs = self.network(inputs)
            loss = self.mixup_criterion(outputs, labels_a, labels_b, lam )
            
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.__clip_gradient(self.optimizer, 0.1)
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            original = y.data
            total += y.size(0)
            correct += (lam * predicted.eq(labels_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())
            train_loss += loss.item()

        epoch_acc = 100. * correct/total
        epoch_loss = train_loss / data.no_batches

        print('Epoch:{} training loss:'
              ' {} training accuracy {}'.format(epoch + 1,
                                               np.round(epoch_loss,4),
                                               np.round(epoch_acc,4)))

        return epoch_acc, epoch_loss
    
    def train_process_features(self, epoch, train_data_engine):
        self.network = self.network.train()
        self.lr_rate_setting(epoch)
        data = train_data_engine

        train_loss = 0
        correct = 0
        total = 0

        print('no of batches', data.no_batches_features)
        for _ in tqdm(range(data.no_batches_features), dynamic_ncols=True):  # range(train.no_batch)
            inputs, labels = data.mini_batch_features()
            # print("2", inputs.shape)
            # print("3", labels.shape)
            labels = labels.long()
            inputs = inputs.to('cuda:0')
            labels = labels.to('cuda:0')
            
            outputs = self.network(inputs.float())
            loss = self.criteria(outputs, labels)
            
            self.optimizer.zero_grad() 
            loss.backward()
            self.__clip_gradient(self.optimizer, 0.1)
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            original = labels.data
            total += labels.size(0)
            correct += predicted.eq(original.data).cpu().sum()
            train_loss += loss.item()

        epoch_acc = 100. * correct/total
        epoch_loss = train_loss / data.no_batches_features

        print('Epoch:{} training loss:'
            ' {} training accuracy {}'.format(epoch + 1,
                                            np.round(epoch_loss,4),
                                            np.round(epoch_acc,4)))

        return epoch_acc, epoch_loss

    def valid_process(self, epoch, valid_data_engine):

        self.network = self.network.eval()
        data = valid_data_engine

        valid_loss = 0
        correct = 0
        total = 0

        data.shuffle_data()
        with torch.no_grad():
            for _ in tqdm(range(data.no_batches), dynamic_ncols=True):  # range(valid.no_batch)
                inputs, labels = data.mini_batch()
                labels = labels.long()
                outputs = self.network(inputs)

                loss = self.criteria(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                original = labels.data
                total += labels.size(0)
                correct += predicted.eq(original.data).cpu().sum()
                valid_loss += loss.detach().item()

        # epoch_acc = torch.true_divide(correct, total).data
        epoch_acc = 100. * correct / total
        epoch_loss = valid_loss / data.no_batches

        print('Epoch:{} '
              'validation loss: '
              '{} validation Accuracy {}'.format(epoch + 1,
                                                 np.round(epoch_loss, 4),
                                                 np.round(epoch_acc, 4)))
        return epoch_acc, epoch_loss
    
    def valid_process_features(self, epoch, valid_data_engine):

        self.network = self.network.eval()
        data = valid_data_engine

        valid_loss = 0
        correct = 0
        total = 0

        # data.shuffle_data()
        with torch.no_grad():
            for _ in tqdm(range(data.no_batches_features), dynamic_ncols=True):  # range(valid.no_batch)
                inputs, labels = data.mini_batch_features()
                labels = torch.FloatTensor(labels).long()
                outputs = self.network(torch.FloatTensor(inputs))

                loss = self.criteria(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                original = labels.data
                total += labels.size(0)
                correct += predicted.eq(original.data).cpu().sum()
                valid_loss += loss.detach().item()

        # epoch_acc = torch.true_divide(correct, total).data
        epoch_acc = 100. * correct / total
        epoch_loss = valid_loss / data.no_batches_features

        print('Epoch:{} '
              'validation loss: '
              '{} validation Accuracy {}'.format(epoch + 1,
                                                 np.round(epoch_loss, 4),
                                                 np.round(epoch_acc, 4)))
        return epoch_acc, epoch_loss

    def load_model_zoo(self, pre_trained_model_path, load_optim=True, epoch_0=False):

        checkpoint = torch.load(pre_trained_model_path)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if epoch_0:
            pretrain_epoch = None
        else:
            pretrain_epoch = checkpoint['epoch']

        pretrain_loss = checkpoint['loss']
        pretrain_valid_acc = checkpoint['acc']

        self.pretrained_epoch = pretrain_epoch
        self.pretrained_valid_acc = pretrain_valid_acc

        print('Loaded Pretrained Epoch:{} loss: {} Accuracy {}'.format(
            pretrain_epoch,
            pretrain_loss,
            pretrain_valid_acc))

    def train_model(self, no_epoch, train_data_engine,
                    valid_data_engine, save_mode = False):

        graph = np.zeros((1, 4))
        if self.pretrained_epoch is None:
            self.pretrained_epoch = 0
        else:
            self.pretrained_epoch +=1    

        for epoch in range(self.pretrained_epoch, (self.pretrained_epoch+no_epoch)):
            #Mixup
            if self.alpha!=0:
                train_acc, train_loss = self.train_process_mixup(epoch, train_data_engine)
            #Feature-based
            elif self.is_baseline:
                train_acc, train_loss = self.train_process_features(epoch, train_data_engine)
            else:
                train_acc, train_loss = self.train_process(epoch, train_data_engine)
                
            if self.is_baseline:
                valid_acc, valid_loss = self.valid_process_features(epoch, valid_data_engine)
            else:
                valid_acc, valid_loss = self.valid_process_features(epoch, valid_data_engine)

            temp = np.array([train_acc, train_loss, valid_acc, valid_loss])
            graph = np.vstack((graph,temp))

            if save_mode is True:
                self.save_model(epoch,valid_acc,valid_loss)

        return graph[1:,:]

    def test_model(self, test_data_engine):

        test_acc, test_loss = self.valid_process(0, test_data_engine)

        return test_acc, test_loss