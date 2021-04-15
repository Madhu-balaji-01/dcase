import torch.nn as nn
# import torch.nn.functional as F
import torch
from torchsummary import summary

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
        self.fc6 = nn.Linear(256, no_class)

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
        out = self.fc6(out)
        return out

if __name__=="__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DCASE_PAST(5)
    model.to(device)
    print(summary(model, (1,512,500)))