import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np

############################################################################
# This class is responsible for a single depth-wise separable convolution step
class dilated_downsample(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding,dilation): 
        super(dilated_downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_in,dilation=dilation),
            nn.Conv2d(C_in, C_out, 1, 1, 0),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# This class is responsible for a single depth-wise separable De-convolution step
class dilated_upsample(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding,dilation): 
        super(dilated_upsample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_in, kernel, stride, padding, groups=C_in,dilation=dilation),
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0)
        )
        self.bn=nn.BatchNorm2d(C_out)
        self.lRelu = nn.LeakyReLU()

    def forward(self,x,last=False):
        x=self.layer(x)
        if last==False:
            x=self.lRelu(self.bn(x))
        return x


#This class incorporates the convolution steps in parallel with different dilation rates and 
#concatenates their output. This class is called at each level of the U-NET encoder 
class BasicBlock_downsample(nn.Module):
    def __init__(self,c1,c2,k1,k2,k3,s1,s2,s3,p1,p2,p3):
        super(BasicBlock_downsample, self).__init__()

        self.d1=dilated_downsample(c1,c2,k1,s1,p1,dilation=3)
        self.d2=dilated_downsample(c1,c2,k2,s2,p2,dilation=2)
        self.d3=dilated_downsample(c1,c2,k3,s3,p3,dilation=1)

    def forward(self,x):
        x1=self.d1(x)
        x2=self.d2(x)
        x3=self.d3(x)

        return torch.cat([x1,x2,x3],dim=1)

#This class incorporates the De-convolution steps in parallel with different dilation rates and 
#concatenates their output. This class is called at each level of the U-NET Decoder 
class BasicBlock_upsample(nn.Module):
    def __init__(self,c1,c2,k1,k2,k3,s1,s2,s3,p1,p2,p3):
        super(BasicBlock_upsample, self).__init__()

        self.d1=dilated_upsample(c1,c2,k1,s1,p1,dilation=3)
        self.d2=dilated_upsample(c1,c2,k2,s2,p2,dilation=2)
        self.d3=dilated_upsample(c1,c2,k3,s3,p3,dilation=1)

        self.resize=dilated_upsample(c2,c2,2,1,0,1)

    def forward(self,x,y=None):
        x1=self.d1(x)
        x2=self.resize(self.d2(x))
        x3=self.d3(x)

        x_result= torch.cat([x1,x2,x3],dim=1)

        if y is not None:
            return torch.cat([x_result,y],dim=1)
        return x_result
    
        
class Dilated_UNET(nn.Module):
    def __init__(self):
        super(Dilated_UNET,self).__init__()

        ##Encoder - with shape output commented for each step
        self.d2=BasicBlock_downsample(3,12,k1=3,k2=4,k3=4,s1=2,s2=2,s3=2,p1=3,p2=3,p3=1) #36,128,128
        self.d3=BasicBlock_downsample(36,36,k1=3,k2=4,k3=4,s1=2,s2=2,s3=2,p1=3,p2=3,p3=1) #108,64,64
        self.d4=BasicBlock_downsample(108,108,k1=3,k2=4,k3=4,s1=2,s2=2,s3=2,p1=3,p2=3,p3=1) #324,32,32
        self.d5=BasicBlock_downsample(324,324,k1=3,k2=4,k3=4,s1=2,s2=2,s3=2,p1=3,p2=3,p3=1) #972,16,16

        ###Decoder - with shape output commented for each step
        self.u1=BasicBlock_upsample(972,108,k1=4,k2=4,k3=4,s1=2,s2=2,s3=2,p1=4,p2=3,p3=1)#324,32,32
        self.u2=BasicBlock_upsample(648,36,k1=4,k2=4,k3=4,s1=2,s2=2,s3=2,p1=4,p2=3,p3=1) #108,64,64
        self.u3=BasicBlock_upsample(216,12,k1=4,k2=4,k3=4,s1=2,s2=2,s3=2,p1=4,p2=3,p3=1) #36,128,128
        self.u4=BasicBlock_upsample(72,4,k1=4,k2=4,k3=4,s1=2,s2=2,s3=2,p1=4,p2=3,p3=1) #12,256,256

        ##Classifier
        self.classifier=nn.Conv2d(12,20,3,1,1) #20,512,1024

        ##Dropout layers
        self.drop1=nn.Dropout2d(p=0.2)
        self.drop2=nn.Dropout2d(p=0.15)
    
    def forward(self,x):
        down2=self.drop1(self.d2(x))
        down3=self.drop1(self.d3(down2))
        down4=self.drop1(self.d4(down3))
        down5=self.drop1(self.d5(down4))

        up1=self.drop2(self.u1(down5,down4))
        up2=self.drop2(self.u2(up1,down3))
        up3=self.drop2(self.u3(up2,down2))
        up4=self.drop2(self.u4(up3))

        return self.classifier(up4)




    












