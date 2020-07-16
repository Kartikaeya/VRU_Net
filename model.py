#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[2]:


class Conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding, bn = True, relu = True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel , stride = stride, padding = padding)
        if bn:
            self.is_bn = True
            self.batchnorm = nn.BatchNorm2d(out_dim)
        if relu:
            self.is_relu = True
            self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.batchnorm(x)
        if self.is_relu:
            x = self.relu(x)
        return x


# In[3]:


class DeConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, in_padding, out_padding = 0, bn = True, relu = True):
        super(DeConv, self).__init__()
        self.convTranspose = nn.ConvTranspose2d(in_dim, out_dim, kernel , stride = stride, padding = in_padding, output_padding = out_padding)
        if bn:
            self.is_bn = True
            self.batchnorm = nn.BatchNorm2d(out_dim)
        if relu:
            self.is_relu = True
            self.relu = nn.ReLU()
    def forward(self , x):
        x = self.convTranspose(x)
        if self.is_bn:
            x = self.batchnorm(x)
        if self.is_relu:
            x = self.relu(x)
        return x


# In[4]:


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        
        self.bb1 = nn.Sequential(
            Conv(3, 16, 7, 1, 3),
            Conv(16, 16, 7, 1, 3),
            Conv(16, 32, 3, 1, 1),
            Conv(32, 64, 3, 2, 1)
        )
        self.bb2 = nn.Sequential(
            Conv(128, 128, 7, 1, 3),
            Conv(128, 128, 7, 1, 3),
            Conv(128, 128, 3, 1, 1),
            Conv(128, 128, 3, 2, 1)
        )
        
        self.bb3 = nn.Sequential(
            Conv(256, 256, 5, 1, 2),
            Conv(256, 256, 5, 1, 2),
            Conv(256, 256, 3, 1, 1),
            Conv(256, 256, 3, 1, 1)
        )
        self.skip1 = nn.Conv2d(3, 64, 3, stride = 2, padding = 1)
        self.skip2 = nn.Sequential(
            Conv(3, 64, 3, stride = 2, padding = 1),
            Conv(64, 128, 3, stride = 2, padding = 1)
        )
        
        self.upsample1 = nn.Sequential(
            DeConv(256, 128, 3, 2, 1, 1),
            Conv(128,128, 3, 1, 1),
            DeConv(128, 64, 3, 2, 1, 1),
            Conv(64,32, 3, 1, 1),
            Conv(32,17, 3, 1, 1)
            
        )
        
        
    def forward(self, x):
        input_shape = x.shape
        x = nn.MaxPool2d(2, 2)(x)
        temp = x
        
        out = self.bb1(x)
        residual1 = self.skip1(temp)
        out = torch.cat((out, residual1), 1)
        
        out = self.bb2(out)
        residual2 = self.skip2(temp)
        out = torch.cat((out, residual2), 1)
        
        out = self.bb3(out)
        out = self.upsample1(out)
        
        out = F.interpolate(out, (input_shape[2], input_shape[3]), mode = 'bilinear')
    
        return out


# In[5]:


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            Conv(in_channels, in_channels, 3, 1, 1),
            Conv(in_channels, in_channels*2, 3, 2, 2),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


# In[6]:


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(    
            Conv(in_channels, in_channels//2, 3, 1, 1),
            Conv(in_channels//2, in_channels//2, 3, 1, 1)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


# In[7]:


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
#         self.img = img
        self.conv1 = Conv(20, 32, 3, 1, 1)
        self.conv2 = nn.ModuleList([Downsample(32*(2**(i-1))) for i in range(1, 5)])
        self.conv3 = nn.ModuleList([DeConv(32*(2**(5-i)),32*(2**(5-i))//2, 2, 2, 1) for i in range(1, 5)])
        self.conv4 = nn.ModuleList([Upsample(32*(2**(5-i))) for i in range(1, 5)])
        self.conv5 = Conv(32, 17, 3, 1, 1)
    def forward(self, x):
#         x = torch.cat((x, self.img), 1)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.conv1(x)
        skip_connections = []
        skip_connections.append(x)
        for i in range(4):
            x = self.conv2[i](x)
            skip_connections.append(x)
        for i in range(4):
            x = self.conv3[i](x)
            target_shape = x.shape
            interpolated_layer = F.interpolate(skip_connections[3-i], (target_shape[2], target_shape[3]), mode = 'bilinear')
            x = torch.cat((x, interpolated_layer), 1)
#             print(x.shape)
            x = self.conv4[i](x)
        x = self.conv5(x)
        return x


# In[8]:


class VRUNet(nn.Module):
    def __init__(self):
        super(VRUNet, self).__init__()
        self.Backbone = BackBone()
        self.Unet_module = nn.ModuleList([Unet() for i in range(3)])
        
    def forward(self, x):
        input_img = x
        confidence_maps_tensor = torch.zeros(x.shape[0], 17*4, x.shape[2], x.shape[3], device = device)
        confidence_maps = []
        x = self.Backbone(x)
        
        confidence_maps.append(x)
        
        for i in range(3):
            x = torch.cat((x, input_img), 1)
            x = self.Unet_module[i](x)
            x = F.interpolate(x, (input_img.shape[2], input_img.shape[3]), mode = 'bilinear')
            confidence_maps.append(x)
            
        for i in range(input_img.shape[0]):
            for j in range(4):
                confidence_maps_tensor[i, 17*j:(17*j + 17), :, :] = confidence_maps[j][i]
        return confidence_maps_tensor


# In[9]:


if __name__ == '__main__':
    complete_model = VRUNet().cuda()

    summary(complete_model, (3, 269, 117))

    # model = Unet(torch.randn(2, 17, 269, 117).cuda()).cuda()

    # model = BackBone().cuda()


# In[ ]:




