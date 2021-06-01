from torch.autograd.grad_mode import F
import torch.nn as nn
# import torch.nn.functional as F
import torch
from torchsummary import summary
import numpy as np

class Conv_Layer(nn.Module):
    def __init__(self,ch_in, ch_out, kernel_size, stride, padding):
        super(Conv_Layer,self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG_M(nn.Module):
    def __init__(self, no_class):
        super(VGG_M,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv_Layer(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.conv2 = Conv_Layer(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = Conv_Layer(256, 384, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))
        self.conv4 = Conv_Layer(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = Conv_Layer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool5 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
        self.fc6 = Conv_Layer(256, 4096, kernel_size=(9, 1), stride=(1, 1), padding=0)
        self.apool6 = nn.AdaptiveAvgPool2d((1,1))
        self.fc7 = nn.Linear(4096,1024)
        self.fc_out = nn.Linear(1024, no_class)

    def features(self,x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.maxpool5(out)
        out = self.fc6(out)
        out = self.apool6(out)
        out = out.view(out.size(0), -1)
        return out

    def classifier(self,x):
        out = self.fc7(x)
        out = self.relu(out)
        out = self.fc_out(out)
        return out

    def forward(self,x):
        out = self.features(x)
        out = self.classifier(out)
        return out

class VGG_M2(nn.Module):
    def __init__(self, no_class):
        super(VGG_M2,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv_Layer(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.conv2 = Conv_Layer(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = Conv_Layer(256, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))
        self.conv4 = Conv_Layer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = Conv_Layer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.maxpool5 = nn.AdaptiveMaxPool2d((1,1))
        self.fc6 = nn.Linear(512, 4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc_out = nn.Linear(4096, no_class)

    def forward(self,x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.maxpool5(out)
        out = out.view(out.size(0), -1)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc_out(out)
        return out
    
# class VGG_M2_mixup(nn.Module):
#     def __init__(self, no_class):
#         super(VGG_M2,self).__init__()
#         self.relu = nn.ReLU()
#         self.conv1 = Conv_Layer(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
#         self.conv2 = Conv_Layer(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
#         self.conv3 = Conv_Layer(256, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))
#         self.conv4 = Conv_Layer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.conv5 = Conv_Layer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         # self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
#         self.maxpool5 = nn.AdaptiveMaxPool2d((1,1))
#         self.fc6 = nn.Linear(512, 4096)
#         self.fc7 = nn.Linear(4096,4096)
#         self.fc_out = nn.Linear(4096, no_class)

#     def mixup(x, shuffle, lam, i, j):
#         if shuffle is not None and lam is not None and i == j:
#             x = lam * x + (1 - lam) * x[shuffle]
#         return x
    
#     def forward(self,x):
#         if isinstance(x, list):
#             x, shuffle, lam = x
#         else:
#             shuffle = None
#             lam = None
        
#         # Decide which layer to mixup
#         j = np.random.randint(15)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.conv1(x)
        
#         x = self.maxpool1(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.conv2(x)
        
#         x = self.maxpool2(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.conv3(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.conv4(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.conv5(x)
        
#         x = self.maxpool5(x)
#         x = x.view(x.size(0), -1)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.fc6(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.relu(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.fc7(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         x = self.relu(x)
        
#         x = self.mixup(x, shuffle, lam, 0, j)
#         out = self.fc_out(x)
#         return out
    
class VGG_M3(nn.Module):
    def __init__(self, no_class):
        super(VGG_M3,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv_Layer(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.conv2 = Conv_Layer(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = Conv_Layer(256, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))
        self.conv4 = Conv_Layer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = Conv_Layer(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.maxpool5 = nn.AdaptiveMaxPool2d((1,1))
        self.fc6 = nn.Linear(512, 4096)
        self.fc7 = nn.Linear(4096,1024)
        self.fc_out = nn.Linear(1024, no_class)

    def forward(self,x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.maxpool5(out)
        out = out.view(out.size(0), -1)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc_out(out)
        return out
class DCASE_PAST(nn.Module):
    def __init__(self, no_class):
        super(DCASE_PAST,self).__init__()
        self.conv1 = Conv_Layer(1, 64, kernel_size=(8,2), stride=(1,1), padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        self.conv2 = Conv_Layer(64, 128, kernel_size=(1,8), stride=(1,1), padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        self.conv3 = Conv_Layer(128, 256, kernel_size=(1,10), stride=(1,1), padding=0)
        self.conv4 = Conv_Layer(256, 512, kernel_size=(10,1), stride=(1,1), padding=0)
        self.apool4 = nn.AdaptiveAvgPool2d((1,1))
        self.fc5 = nn.Linear(512, 256)
        self.drop5 = nn.Dropout2d()
        self.fc_out = nn.Linear(256, no_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.apool4(out)
        out = out.view(out.size(0), -1)
        out = self.fc5(out)
        out = self.drop5(out)
        out = self.fc_out(out)
        return out

class DCASE_PAST2(nn.Module):
    def __init__(self, no_class):
        super(DCASE_PAST2,self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = Conv_Layer(1, 64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv2 = Conv_Layer(64, 128, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv3 = Conv_Layer(128, 256, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv4 = Conv_Layer(256, 512, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv5 = Conv_Layer(512, 512, kernel_size=(3,3), stride=(1,1), padding=0)
        self.glbmaxpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc6 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, no_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.maxpool(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.glbmaxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc_out(out)
        return out

class Baseline_Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Baseline_Block, self).__init__()
        self.fc = nn.Linear(in_features=ch_in, out_features=ch_out)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

class BASELINE(nn.Module):
    def __init__(self,no_class):
        super(BASELINE, self).__init__()
        self.model = nn.Sequential(
          nn.Linear(512,512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(512,128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=0.3),

          nn.Linear(128,64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=0.3),

          nn.Linear(64,no_class)

        )
        
        
        # self.fc1 = Baseline_Block(512, 512)
        # self.fc2 = Baseline_Block(512, 128)
        # self.fc3 = Baseline_Block(128, 64)
        # self.fc_out = nn.Linear(64, no_class)
    
    def forward(self, x):
        # out = self.fc1(x)
        # out = self.fc2(out)
        # out = self.fc3(out)
        # out = self.fc_out(out)
        out = self.model(x)
        return out

class ENSEMBLE(nn.Module):
    def __init__(self, model_a, model_b, no_class):
        super(ENSEMBLE, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        
        self.model_a.fc_out = nn.Identity()
        self.model_b.fc_out = nn.Identity()
        
        self.relu = nn.ReLU()
        
        self.ensemble = nn.Linear(4096+256, no_class)
    
    def forward(self, x):
        x1 = self.model_a(x.clone())
        x1 = x1.view(x1.size(0), -1)
        x2 = self.model_b(x)
        x2 = x2.view(x2.size(0), -1)
        out = torch.cat((x1,x2), dim=1)
        out = self.relu(out)
        out = self.ensemble(out)
        
        return out

class ENSEMBLE_BASELINE(nn.Module):
    def __init__(self, model_a, model_b, no_class):
        super(ENSEMBLE_BASELINE,self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.model_a.fc_out = nn.Identity()
        self.model_b.fc_out = nn.Identity()
        self.relu = nn.ReLU()
        self.ensemble = nn.Linear(4096+256, no_class)
    
    def forward(self, x):

        x1 = self.model_a(x)
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.model_b(x)
        x2 = x2.view(x2.size(0), -1)
        
        out = torch.cat((x1,x2), dim=1)
        out = self.relu(out)
        out = self.ensemble(out)
        
        return out
    


    
        
        
    
if __name__=="__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_class = 10
    # model_a = VGG_M(10)
    # model_b = DCASE_PAST(10)
    # model=ENSEMBLE(model_a, model_b, 10)
    model = BASELINE(no_class)
    model.to(device)
    print(summary(model))