import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np


class conv_bn_LRelu(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding): 
        super(conv_bn_LRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x

class convTranspose_bn_LRelu(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding): 
        super(convTranspose_bn_LRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x

class MobileNetV2_seg(nn.Module):
    def __init__(self, pretrained): 
        super(MobileNetV2_seg, self).__init__()
        self.preprocess = nn.Sequential(
            conv_bn_LRelu(3, 3, (4,4), (2,4), (1,1)), # b, 16, 256, 256
            # conv_bn_LRelu(8, 3, 4, 2, 1), # b, 16, 256, 256
        )

        for param in pretrained.named_parameters():
            param[1].requires_grad = True

        pretrained[0][18] = nn.Identity()
        self.encoder = pretrained

        self.decoder = nn.Sequential(
            convTranspose_bn_LRelu(320, 64, 4, 4, 0), # b, 64, 32, 32
            convTranspose_bn_LRelu(64, 34, 4, 4, 0), # b, 32, 128, 128
            # convTranspose_bn_LRelu(32, 34, (4,8), (4,8), 0), # b, 34, 512, 1024
            nn.ConvTranspose2d(34, 34, (4,8), (4,8), (0,0), groups=34) # b, 34, 1024, 2048
        )

    def forward(self, x):
        # x -> (b, 3, 1024, 2048)
        x = self.preprocess(x)  # (b, 3, 256, 256)
        x = F.interpolate(x, (224, 224), mode='bilinear') # (b, 3, 256, 256)
        x = self.encoder(x) # (b, 320, 7, 7)
        # print(f"Encoder: {x.shape}")
        x = F.interpolate(x, (8, 8), mode='bilinear') # (b, 3, 8, 8)
        # print(f"Interp: {x.shape}")
        x = self.decoder(x) # (b, 34, 1024, 2048)
        # print(f"Decoder: {x.shape}")

        return x

class ResNet18_seg(nn.Module):
    def __init__(self, pretrained): 
        super(ResNet18_seg, self).__init__()
        self.preprocess = nn.Sequential(
            conv_bn_LRelu(3, 3, (4,4), (2,4), (1,1)), # b, 16, 256, 256
            # conv_bn_LRelu(8, 3, 4, 2, 1), # b, 16, 256, 256
        )

        for param in pretrained.named_parameters():
            param[1].requires_grad = True

        # pretrained[0][18] = nn.Identity()
        self.encoder = pretrained

        self.decoder = nn.Sequential(
            convTranspose_bn_LRelu(512, 64, 4, 4, 0), # b, 64, 32, 32
            convTranspose_bn_LRelu(64, 34, 4, 4, 0), # b, 32, 128, 128
            # convTranspose_bn_LRelu(32, 34, (4,8), (4,8), 0), # b, 34, 512, 1024
            nn.ConvTranspose2d(34, 34, (4,8), (4,8), (0,0), groups=34) # b, 34, 1024, 2048
        )

    def forward(self, x):
        # x -> (b, 3, 1024, 2048)
        x = self.preprocess(x)  # (b, 3, 256, 256)
        x = F.interpolate(x, (224, 224), mode='bilinear') # (b, 3, 256, 256)
        x = self.encoder(x) # (b, 320, 7, 7)
        # print(f"Encoder: {x.shape}")
        x = F.interpolate(x, (8, 8), mode='bilinear') # (b, 3, 8, 8)
        # print(f"Interp: {x.shape}")
        x = self.decoder(x) # (b, 34, 1024, 2048)
        # print(f"Decoder: {x.shape}")

        return x

#############U-Net########        
    
class Downsample(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding): 
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.Conv2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding): 
        super(Upsample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel, stride, padding, groups=C_in),
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0)
        )
        self.bn=nn.BatchNorm2d(C_out)
        self.lRelu = nn.LeakyReLU()

    def forward(self,x1,x2=None,last=False):
        if x2 is None:
            x=self.layer(x1)
        else:
            x=torch.cat((x1,x2),dim=1)
        
        if last==False:
            x=self.lRelu(self.bn(x))
        return x



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        #encoder
        self.d1=Downsample(3,16, (4,4),(2,4),(1,0)) #16, 256, 256
        self.d2=Downsample(16,32, 4, 2, 1) #32, 128, 128
        self.d3=Downsample(32,64, 4, 2, 1) #64, 64, 64
        self.d4=Downsample(64,128, 4, 2, 1) #128, 32, 32
        self.d5=Downsample(128,256, 4, 2, 1) #256, 16, 16
        self.d6=Downsample(256,512, 4, 2, 1) #256, 8, 8

        self.u1=convTranspose_bn_LRelu(512,256, 4, 2, 1) #128, 16, 16
        self.u2=Upsample(512,128, 4, 2, 1) # 128, 32, 32
        self.u3=Upsample(256,64, 4, 2, 1)#64, 64 ,64
        self.u4=Upsample(128,32, 4, 2, 1) #32, 128,128
        self.u5=Upsample(64,16, 4, 2, 1) #16, 256, 256
        self.u6=Upsample(32,20,(4,4),(2,4),(1,0)) #34, 512,1024

        self.drop1=nn.Dropout2d(p=0.2)
        self.drop2=nn.Dropout2d(p=0.15)

    def forward(self,x):

        down1=self.d1(x)
        down2=self.drop1(self.d2(down1))
        
        down3=self.d3(down2)
        down4=self.drop1(self.d4(down3))

        down5=self.d5(down4)
        down6=self.drop1(self.d6(down5))


        up1=self.u1(down6)
        up2=self.drop2(self.u2(down5,up1))
        
        up3=self.u3(down4,up2)
        up4=self.drop2(self.u4(down3,up3))

        up5=self.u5(down2,up4)
        up6=self.u6(down1,up5,last=True)

        return up6

