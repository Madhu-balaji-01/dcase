import torch.nn as nn
import torch.nn.functional as F
import torch
from data_engin import Data_Engin

# class Conv_Layer(nn.Module):
#     def __init__(self,ch_in, ch_out, kernel_size, stride, padding):
#         super(Conv_Layer,self).__init__()
#         self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,
#                               stride=stride, padding=padding)
#         self.bn = nn.BatchNorm2d(ch_out)
#         self.relu = nn.ReLU()
#
#     def forward(self,x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class VGG_M(nn.Module):
#
#     def __init__(self, no_class):
#         super(VGG_M,self).__init__()
#         self.conv1 = Conv_Layer(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
#         self.conv2 = Conv_Layer(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
#         self.conv3 = Conv_Layer(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))
#         self.conv4 = Conv_Layer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.conv5 = Conv_Layer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.maxpool5 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
#         self.conv6 = Conv_Layer(256, 4096, kernel_size=(9, 1), stride=(1, 1), padding=0)
#         self.apool6 = nn.AvgPool2d(kernel_size=(1,11), stride=(1,1))
#         self.fc7 = nn.Linear(20480,1024)
#         self.fc8 = nn.Linear(1024, no_class)
#
#     def forward(self,x):
#         out = self.conv1(x)
#         out = self.maxpool1(out)
#         out = self.conv2(out)
#         out = self.maxpool2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#         out = self.maxpool5(out)
#         out = self.conv6(out)
#         out = self.apool6(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc7(out)
#         out = F.relu(out)
#         out = F.dropout(out, p=0.5, training=self.training)
#         out = self.fc8(out)
#
#         return out


class VGGish_Net(nn.Module):
    def __init__(self, n_classes):
        super(VGGish_Net, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.batch_N_C1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch_N_C2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.batch_N_C3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.batch_N_C4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.batch_N_C5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batch_N_C6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batch_N_C7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batch_N_C8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batch_N_C9 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512*3*4, 1024)
        self.batch_N_f1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.batch_N_f2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, self.n_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_N_C1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch_N_C2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batch_N_C3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batch_N_C4(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.batch_N_C5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batch_N_C6(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.batch_N_C7(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv8(x)
        x = self.batch_N_C8(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv9(x)
        x = self.batch_N_C9(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512*3*4)

        x = self.fc1(x)
        x = self.batch_N_f1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batch_N_f2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # x = F.relu(x)

        return x


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = Data_Engin(address='./dataset/train_temp.csv', spectra_type='Spectrum',
                      device=device, batch_size=32)

    from models.vgg_m import VGG_M

    net = VGG_M(5)
    net = net.to(device=device, dtype=torch.float32)
    x, y = train.mini_batch()
    output = net(x)