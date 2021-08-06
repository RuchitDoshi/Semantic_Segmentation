import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torchvision
from tqdm import tqdm
from utils import Visualize
import matplotlib.pyplot as plt

## TODO: [*] Mean IOU
## TODO: [*] Visualize -> can write a separate visualize function

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training_onehot(model,train_loader,criterion,optimizer):
    '''
    Training one epoch of the model 

    return: Training loss of one epoch
    '''
    model.train()
    avg_loss=[]

    for batch_num, (feats, labels,targets) in tqdm(enumerate(train_loader), desc="Iteration num "):
        feats, labels,targets = feats.to(device), labels.to(device),targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(feats)
        # print(outputs.dtype)
        # print(torch.argmax(outputs, dim=1, keepdim=False).dtype)
        # print(labels.float().dtype)
        # print(labels.unsqueeze(1).shape)
        # loss=criterion(torch.argmax(outputs, dim=1, keepdim=False).float(),labels)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())
            
        torch.cuda.empty_cache()
        del feats
        del labels
        del loss

    print('Train Loss: {:.6f}'.format(sum(avg_loss)/len(avg_loss)))
    return sum(avg_loss)/len(avg_loss)

def validation_onehot(model,test_loader,criterion):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    avg_loss=[]
    avg_IOU=[]
    out=[]
    inp=[]
    
    for batch_num, (feats, labels,targets) in enumerate(test_loader):
        feats, labels,targets = feats.to(device), labels.to(device),targets.to(device)

        outputs=model(feats)
        # outputs=outputs.argmax(axis=1)
        temp=outputs.argmax(axis=1).squeeze().detach().cpu().numpy()
        inp=feats.squeeze().detach().cpu().numpy()
        # print(temp.shape)
        # print(temp.squeeze().shape)
        # out.append(temp)
        loss=criterion(outputs,targets)
        m_iou = Visualize.iou(temp, labels.detach().cpu().numpy())
        avg_loss.append(loss.item())
        avg_IOU.append(m_iou)
        # del feats
        # del temp
        # del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))
    print(f'Mean IOU: {Visualize.iou(temp, labels.detach().cpu().numpy())}')
    print(f'Avg. Mean IOU: {sum(avg_IOU)/len(avg_IOU):.4f}')

    del feats
    # del temp
    del labels

    return np.array(inp), np.array(temp)


def training(model,train_loader,criterion,optimizer):
    '''
    Training one epoch of the model 

    return: Training loss of one epoch
    '''
    model.train()
    avg_loss=[]

    for batch_num, (feats, labels) in tqdm(enumerate(train_loader), desc="Iteration num "):
        feats, labels = feats.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(feats)
        # print(outputs.dtype)
        # print(torch.argmax(outputs, dim=1, keepdim=False).dtype)
        # print(labels.float().dtype)
        # print(labels.unsqueeze(1).shape)
        # loss=criterion(torch.argmax(outputs, dim=1, keepdim=False).float(),labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())
            
        torch.cuda.empty_cache()
        del feats
        del labels
        del loss

    print('Train Loss: {:.6f}'.format(sum(avg_loss)/len(avg_loss)))
    return sum(avg_loss)/len(avg_loss)

def validation(model,test_loader,criterion):
    '''
    Validation for  one epoch of the model  

    return: Validation loss of one epoch
    '''
    model.eval()
    avg_loss=[]
    avg_IOU=[]
    out=[]
    inp=[]
    
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)

        outputs=model(feats)
        # outputs=outputs.argmax(axis=1)
        temp=outputs.argmax(axis=1).squeeze().detach().cpu().numpy()
        inp=labels.detach().cpu().numpy()
        # print(temp.shape)
        # print(temp.squeeze().shape)
        # out.append(temp)
        loss=criterion(outputs,labels)
        m_iou = Visualize.iou(temp, labels.detach().cpu().numpy())
        avg_loss.append(loss.item())
        avg_IOU.append(m_iou)
        # del feats
        # del temp
        # del labels
    model.train()
    print('Validation Loss: {:.4f}'.format(sum(avg_loss)/len(avg_loss)))
    print(f'Mean IOU: {Visualize.iou(temp, labels.detach().cpu().numpy())}')
    print(f'Avg. Mean IOU: {sum(avg_IOU)/len(avg_IOU):.4f}')

    del feats
    # del temp
    del labels

    return np.array(inp), np.array(temp)