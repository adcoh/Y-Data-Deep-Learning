import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 


class network1(nn.Module):
    """
    Simple convolutional neural network 
    """
    def __init__(self):
        super(network1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
#         self.upsample1 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.conv3 = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
#         print('input shape: ', x.shape)
        x = F.leaky_relu(self.conv1(x))
#         print('1st layer output shape: ', x.shape)
        x = F.leaky_relu(self.conv2(x))
#         print('2nd layer output shape: ', x.shape)
        x = self.upsample1(x, output_size=(144, 144))
#         print('Upsample layer output shape: ', x.shape)
        x = F.leaky_relu(self.conv3(x))
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
        x = F.leaky_relu(self.conv1(x))
#         print('1st layer output shape: ', x.shape)
        x = F.leaky_relu(self.conv2(x))
#         print('2nd layer output shape: ', x.shape)
        x = self.upsample1(x, output_size=(h*2, w*2))
        h, w = x.shape[-2:]
#         print('Upsample layer output shape: ', x.shape)
        x_mid = F.leaky_relu(self.conv3_1(x))
        # print('final layer x_mid output shape: ', x_mid.shape)
        x_large = self.upsample2(x, output_size=(h*2, w*2))
        x_large = F.leaky_relu(self.conv3_2(x_large))
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
        x = F.leaky_relu(self.conv1(x))
        # print('1st layer output shape: ', x.shape)
        x_res = self.res_block1_2(self.res_block1_1(x))
        x = F.leaky_relu(x + x_res)
        # print('1st residual layer output shape: ', x.shape)
        x_res = self.res_block2_2(self.res_block2_1(x))
        x = F.leaky_relu(x_res + x)
        # print('2nd residual layer output shape: ', x.shape)
        x = self.upsample1(x, output_size=(h*2, w*2))
        h, w = x.shape[-2:]
        # print('1st Upsample layer output shape: ', x.shape)
        x_mid = F.leaky_relu(self.conv3_1(x))
        # print('final layer x_mid output shape: ', x_mid.shape)
        x_res = self.res_block3_2(self.res_block3_1(x))
        x = F.leaky_relu(x_res + x)
        x_large = self.upsample2(x, output_size=(h*2, w*2))
        # print('final layer x_large output shape: ', x_large.shape)
        x_large = F.leaky_relu(self.conv3_2(x_large))
        return x_mid, x_large


class network4(nn.Module):
    """ 
        Network outputs images of 2x and 4x of input size, 
        convolutions through dilated or "Atrous" convolution blocks
    """
    def __init__(self):
        super(network4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.dil_block1_1 = nn.Conv2d(32,32, 3, padding=1, dilation=1)
        self.dil_block1_2 = nn.Conv2d(32,32, 3, padding=2, dilation=2)
        self.dil_block1_3 = nn.Conv2d(32,32, 3, padding=4, dilation=4)
        self.dil_block1_4 = nn.Conv2d(96,32, 3, padding=1)
        self.dil_block2_1 = nn.Conv2d(32,32, 3, padding=1, dilation=1)
        self.dil_block2_2 = nn.Conv2d(32,32, 3, padding=2, dilation=2)
        self.dil_block2_3 = nn.Conv2d(32,32, 3, padding=4, dilation=4)
        self.dil_block2_4 = nn.Conv2d(96,32, 3, padding=1)
        self.dil_block3_1 = nn.Conv2d(32,32, 3, padding=1, dilation=1)
        self.dil_block3_2 = nn.Conv2d(32,32, 3, padding=2, dilation=2)
        self.dil_block3_3 = nn.Conv2d(32,32, 3, padding=4, dilation=4,)
        self.dil_block3_4 = nn.Conv2d(96,32, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(32, 3, 1)
        self.conv3_2 = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.leaky_relu(self.conv1(x))
        # print('1st layer output shape: ', x.shape)
        x1, x2, x3 = self.dil_block1_1(x), self.dil_block1_2(x), self.dil_block1_3(x)
        # print('1st dilation output shape: ', x1.shape)
        # print('2nd dilation output shape: ', x2.shape)
        # print('3rd dilation output shape: ', x3.shape)
        x = self.dil_block1_4(F.leaky_relu(torch.cat([x1, x2, x3], dim=1)))
        # print('1st dilation layer output shape: ', x.shape)
        x1, x2, x3 = self.dil_block2_1(x), self.dil_block2_2(x), self.dil_block2_3(x)
        x = self.dil_block2_4(F.leaky_relu(torch.cat([x1, x2, x3], dim=1)))
        # print('2nd dilation layer output shape: ', x.shape)
        x = self.upsample1(x, output_size=(h*2, w*2))
        h, w = x.shape[-2:]
        # print('1st Upsample layer output shape: ', x.shape)
        x_mid = F.leaky_relu(self.conv3_1(x))
        # print('final layer x_mid output shape: ', x_mid.shape)
        x1, x2, x3 = self.dil_block3_1(x), self.dil_block3_2(x), self.dil_block3_3(x)
        x = self.dil_block3_4(F.leaky_relu(torch.cat([x1, x2, x3], dim=1)))
        x_large = self.upsample2(x, output_size=(h*2, w*2))
        # print('final layer x_large output shape: ', x_large.shape)
        x_large = F.leaky_relu(self.conv3_2(x_large))
        return x_mid, x_large


class network5(nn.Module):
    
    def __init__(self):
        super(network5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 3, 1)
        self.backbone = models.vgg16(pretrained=True)._modules['features'][:4]
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.mean = torch.Tensor([0.485,0.456, 406]).reshape(1,3,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)

    def forward(self, x):
        h, w = x.shape[-2:]
        out = self.backbone(x)
        # print('backbone layer output shape: ', out.shape)
        x = F.leaky_relu(self.conv1(x))
        # print('1st layer output shape: ', x.shape)
        x = F.leaky_relu(self.conv2(x))
        # print('2nd layer output shape: ', x.shape)
        x = F.leaky_relu(torch.cat([x, out], dim=1))
        # print('Concatenated output shape: ', x.shape)
        x = self.upsample1(x, output_size=(h*2, w*2))
        # print('1st upsample layer output shape: ', x.shape)
        h, w = x.shape[-2:]
        x_mid = F.leaky_relu(self.conv3(x))
        # print('Mid_output shape: ', x_mid.shape)
        x = F.leaky_relu(self.conv4(x))
        # print('4th conv layer output shape: ', x.shape)
        x_large = self.upsample2(x, output_size=(h*2, w*2))
        # print('2nd upsample layer output shape: ', x_large.shape)
        x_large = F.leaky_relu(self.conv5(x_large))
        # print('x_large output shape: ', x_large.shape)
        return x_mid, x_large


class network6(network5):
    
    def __init__(self):
        super(network6, self).__init__()
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 192, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, 1)
        
    def forward(self, x):
        out = self.backbone(x)
        # print('backbone layer output shape: ', out.shape)
        x = F.leaky_relu(self.conv1(x))
        # print('1st layer output shape: ', x.shape)
        x = F.leaky_relu(self.conv2_1(x))
        # print('2_1 layer output shape: ', x.shape)
        x = F.leaky_relu(self.conv2_2(x))
        # print('2_2 layer output shape: ', x.shape)
        x = F.leaky_relu(torch.cat([x, out], dim=1))
        # print('Concatenated output shape: ', x.shape)
        x = F.pixel_shuffle(x, upscale_factor=2)
        # print('1st upsample layer output shape: ', x.shape)
        x_mid = F.leaky_relu(self.conv3(x))
        # print('Mid_output shape: ', x_mid.shape)
        x = F.leaky_relu(self.conv4(x))
        # print('4th conv layer output shape: ', x.shape)
        x_large = F.pixel_shuffle(x, upscale_factor=2)
        # print('2nd upsample layer output shape: ', x_large.shape)
        x_large = F.leaky_relu(self.conv5(x_large))
        # print('x_large output shape: ', x_large.shape)
        return x_mid, x_large