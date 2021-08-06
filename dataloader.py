import torch 
import torchvision
import torch.utils.data
from skimage import io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import PIL
from PIL import Image


class MyDataset_onehot(torch.utils.data.Dataset):
    def __init__(self,csv,transform=None,num_class=34):
        self.path=pd.read_csv(csv)
        self.num_class=num_class
        self.mean=np.array([[[0.485]],[[0.456]],[[0.406]]])
        self.std= np.array([[[0.229]], [[0.224]], [[0.225]]])

        
        self.transform=transform

    def __len__(self):
        return len(self.path.iloc[:,0])

    def __getitem__(self,index):
        X = Image.open(self.path.iloc[index,0]).convert('RGB')
        Y = Image.open(self.path.iloc[index,1])

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        X = np.asarray(X)[::2,::2,:]
        Y = np.asarray(Y)[::2,::2]

        X=np.transpose(X,(2,0,1))
        X=X/255.
        X=(X-self.mean)/self.std
    
        X=torch.from_numpy(X).float() 
        Y=torch.from_numpy(Y).long() 

        target=np.zeros((self.num_class,Y.shape[0],Y.shape[1]))
        for i in range(self.num_class):
            target[i][Y==i]=1 
        return X,Y,target



class MyDataset(torch.utils.data.Dataset):
    def __init__(self,csv,transform=None,num_class=19):
        self.path=pd.read_csv(csv)
        self.num_class=num_class
        self.mean=np.array([[[0.485]],[[0.456]],[[0.406]]])
        self.std= np.array([[[0.229]], [[0.224]], [[0.225]]])
        self.ignore_label=19

        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                            3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                            7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                            14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                            18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                            28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}

        
        self.transform=transform

    def __len__(self):
        return len(self.path.iloc[:,0])

    def __getitem__(self,index):
        X = Image.open(self.path.iloc[index,0]).convert('RGB')
        Y = Image.open(self.path.iloc[index,1])

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        X = np.asarray(X)[::2,::2,:]
        Y = np.asarray(Y)[::2,::2]

        mask = np.zeros(Y.shape)
        for k, v in self.id_to_trainid.items():
            mask[Y == k] = v

        X=np.transpose(X,(2,0,1))
        X=X/255.
        X=(X-self.mean)/self.std
    
        X=torch.from_numpy(X).float() 
        mask=torch.from_numpy(mask).long() 

        return X,mask


    