import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import torchvision
import warnings
import pdb
from dataloader import MyDataset
from model import Dilated_UNET
from train import training, validation, training_onehot,validation_onehot
from utils import data_csv, diceLoss, init_weights

if __name__ == '__main__':

    #arguments for num_epochs and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='arg1', type=int, help="Number of Epochs")
    parser.add_argument(dest='arg2', type=int, default=16, help="Batch Size")

    args = parser.parse_args()
    num_epochs = args.arg1
    batch_size = args.arg2

    print(num_epochs, batch_size)

    #Making folders to save reconstructed images, input images and weights
    if not os.path.exists("output"):
        os.mkdir("output")

    if not os.path.exists("input"):
        os.mkdir("input")

    if not os.path.exists("weights"):
        os.mkdir("weights")

    # if not os.path.exists("data/train_data.csv"):
    #     root='data/'
    #     for split in ['train/','val/','test/']:
    #         data_csv(root,split)
    
    warnings.filterwarnings('ignore')


    ##transformer characteristics
    augment = [
    transforms.RandomRotation((0,10)),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    transforms.RandomCrop(256)
    ]
    tfs = transforms.Compose(augment)


    ##Train Loader
    train_dataset = MyDataset('data/train_data.csv',transform=None)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    # print(len(train_loader))
    
    #val data_loader
    validation_dataset = MyDataset('data/val_data.csv',transform=None)
    val_loader_args = dict(batch_size=1, shuffle=False, num_workers=4)
    val_loader = data.DataLoader(validation_dataset, **val_loader_args)

    #Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # MODEL instance and xavier initialization of the weights
    model= Dilated_UNET()
    model.apply(init_weights)
    model=model.to(device)

    #Optimizer, criterion and scheduler
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    criterion=nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                    factor=0.1, patience=5, verbose=False, 
                    threshold=1e-3, threshold_mode='rel', 
                        cooldown=5, min_lr=1e-5, eps=1e-08)
    
    Path='weights/100_t.pth'
    model.load_state_dict(torch.load(Path))
    print(optimizer)

    inp, output=validation(model,val_loader,criterion)
    name='outputs/final_out.npy' 
    name_in='inputs/final_label.npy'       
    np.save(name,output)
    del output
    np.save(name_in,inp)
    del inp


    
    # # Num epochs for Training and Validation functions
    # for epoch in range(num_epochs):
    #     start_time=time.time()
    #     print('Epoch no: ',epoch)
    #     train_loss = training(model,train_loader,criterion,optimizer)
        
    #     #Saving weights after every 20epochs
    #     if epoch%10==0:
    #         inp, output=validation(model,val_loader,criterion)
    #         name='output/'+str(epoch) +'.npy' 
    #         name_in='input/'+str(epoch) +'.npy'       
    #         np.save(name,output)
    #         del output
    #         np.save(name_in,inp)
    #         del inp

    #     if epoch%10==0:
    #         path='weights/'+ str(epoch) +'_t.pth'
    #         torch.save(model.state_dict(),path)
    #         print(optimizer)
        
    #     scheduler.step(train_loss)
    #     print("Time : ",time.time()-start_time)
    #     print('='*50)

