import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

#utility functions
## TODO: [*] utils.py shouldn't be run as main. Add this to the main.py script

def data_csv(root,split):
    img_base=os.path.join(root,'leftImg8bit',split)
    label_base=os.path.join(root,'gtFine',split)
    
    img=[]
    for folder in os.listdir(img_base):
        for file in os.listdir(os.path.join(img_base+ folder + '/')):
            if file.endswith('.png'):
                img.append(img_base+folder+'/'+ file)
    
    label=[]
    for i in img:
        img_path = i.rstrip()
        lbl_path = os.path.join(label_base,i.split(os.sep)[-2],
                            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png")
        label.append(lbl_path)
    
    d={'image':img,'label':label}
    data=pd.DataFrame(data=d)
    
    data.to_csv(root+split[:-1]+'_data.csv',index=False)

def conv2D_shape(input_shape, kernel, stride, padding=(0,0), dilation=(1,1)):
    '''
    Function to help in convolution shape calculations. Eg. print(conv2D_shape((1024, 2048), (4,4), (2,4), (1,1), (1,1)))
    '''
    h_in, w_in = input_shape
    h_out = np.floor(((h_in + 2*padding[0]-dilation[0]*(kernel[0]-1)-1)/stride[0]) + 1)
    w_out = np.floor(((w_in + 2*padding[1]-dilation[1]*(kernel[1]-1)-1)/stride[1]) + 1)

    return (h_out, w_out)


class Visualize:

    # void_classes=[0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    # valid_classes=[7, 8,11,12, 13, 17,19,20,21,22,23,24,25,26, 27,28,31,32,33]
    colors= [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0,0,0]
    ]
    num_class=20
    
    @classmethod
    def visualize(cls, pred):
        output=np.zeros((pred.shape[0],pred.shape[1],3),dtype=int)
        for i in range(cls.num_class):
            output[np.where(pred==i)]=cls.colors[i]   
        return output

    @classmethod
    def iou(cls, pred, labels):
        unions=[]
        intersections=[]
        for clas in range(cls.num_class):
            TP = ((pred==clas) & (labels==clas)).sum()
            FP = ((pred==clas) & (labels!=clas)).sum()
            FN = ((pred!=clas) & (labels==clas)).sum()
            # Complete this function
            intersection = TP
            union = (TP+FP+FN)
            if union == 0:
                intersections.append(0)
                unions.append(0)
                # if there is no ground truth, do not include in evaluation
            else:
                intersections.append(intersection)
                unions.append(union)
                # Append the calculated IoU to the list ious
        mean_iou=sum(intersections)/sum(unions)
        return mean_iou



class diceLoss(torch.nn.Module):
    def __init__(self):
        super(diceLoss, self).__init__()
        
    def forward(self, output, target):
        return self.helper(output, target)
    
    def helper(self,output,target):
        output = F.softmax(output, 1)
        numerator = 2 * torch.sum(output * target)
        denominator = torch.sum(output + target)
        return 1 - (numerator + 1) / (denominator + 1)

def init_weights(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)) and m.weight.requires_grad == True:
        # print(m.type, m.weight.requires_grad)
        torch.nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0)