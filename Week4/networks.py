import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 


class network1(nn.Module):
    
    def __init__(self):
        super(network1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
#         self.upsample1 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv3 = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
#         print('input shape: ', x.shape)
        x = self.conv1(x)
#         print('1st layer output shape: ', x.shape)
        x = self.conv2(F.leaky_relu(x))
#         print('2nd layer output shape: ', x.shape)
        x = self.upsample1(F.leaky_relu(x), output_size=(144, 144))
#         print('Upsample layer output shape: ', x.shape)
        x = self.conv3(F.leaky_relu(x))
#         print('final layer output shape: ', x.shape)
        return x



class network2(network1):
    
    def __init__(self):
        super(network2, self).__init__()
        self.upsample2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(64, 3, 1)
        self.conv3_2 = nn.Conv2d(64, 3, 1)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv1(x)
#         print('1st layer output shape: ', x.shape)
        x = self.conv2(F.leaky_relu(x))
#         print('2nd layer output shape: ', x.shape)
        x = self.upsample1(x, output_size=(h*2, w*2))
        h, w = x.shape[-2:]
#         print('Upsample layer output shape: ', x.shape)
        x = F.leaky_relu(x)
        x_mid = self.conv3_1(x)
        # print('final layer x_mid output shape: ', x_mid.shape)
        x_large = self.upsample2(x, output_size=(h*2, w*2))
        x_large = self.conv3_2(F.leaky_relu(x_large))
        # print('final layer x_large output shape: ', x_large.shape)
        return x_mid, x_large


class network3(nn.Module):
    
    def __init__(self):
        super(network3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.res_block1_1 = nn.Conv2d(32,32, 3, padding=1)
        self.res_block1_2 = nn.Conv2d(32,32, 3, padding=1)
        self.res_block2_1 = nn.Conv2d(32,32, 3, padding=1)
        self.res_block2_2 = nn.Conv2d(32,32, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.res_block3_1 = nn.Conv2d(32,32, 3, padding=1)
        self.res_block3_2 = nn.Conv2d(32,32, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(32, 3, 1)
        self.conv3_2 = nn.Conv2d(32, 3, 1)

    def forward(self, x):
#         print('input shape: ', x.shape)
        h, w = x.shape[-2:]
        x = self.conv1(x)
        # print('1st layer output shape: ', x.shape)
        x_res = self.res_block1_1(x)
        x_res = self.res_block1_2(x_res)
        x = F.leaky_relu(x + x_res)
        # print('1st residual layer output shape: ', x.shape)
        x_res = self.res_block2_1(x)
        x_res = self.res_block2_2(x_res)
        x = F.leaky_relu(x_res + x)
        # print('2nd residual layer output shape: ', x.shape)
        x = self.upsample1(x, output_size=(h*2, w*2))
        h, w = x.shape[-2:]
        # print('1st Upsample layer output shape: ', x.shape)
        x_mid = self.conv3_1(F.leaky_relu(x))
        # print('final layer x_mid output shape: ', x_mid.shape)
        x_res = self.res_block3_1(x)
        x_res = self.res_block3_2(x_res)
        x = F.leaky_relu(x_res + x)
        x_large = self.upsample2(x, output_size=(h*2, w*2))
        # print('final layer x_large output shape: ', x_large.shape)
        x_large = self.conv3_2(F.leaky_relu(x_large))
        return x_mid, x_large