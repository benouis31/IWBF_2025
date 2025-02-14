import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import time
import torch.optim.lr_scheduler as lr_scheduler
from sklearn import model_selection

from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, train_mode, device, mask_ratio=0.8, num_masked=20):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_contrastive_loss = 0
    epoch_constrain_loss = 0
    print('*****************training*******************')
    start = time.time()
    train_loader = tqdm(train_loader, total=len(train_loader), unit="batch")

    for batch_idx, data in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        model.zero_grad()
        #print(data[0].shape)
        
        x = data[0].float().to(device)
        #print(x.shape)
        eda=x[:, :, 0]
        #print(eda.shape)
        bvp=x[:, :, 1]
            #bvp = sample_batched["bvp"].type(torch.float32)
            #temp = sample_batched["temp"].type(torch.float32)
        temp=x[:, :, 2]
        y = data[1].long().to(device)
            
        if train_mode in ['supervised', 'frozen', 'fine-tuned']:
            #loss, pred = model.supervised_train_forward(x, y)
            loss, pred = model.supervised_train_forward(eda,bvp,temp, y)
        elif train_mode == 'ssl':
            #loss, [contrastive_loss, constrain_loss] = model.ssl_train_forward(x, mask_ratio=mask_ratio, num_masked=num_masked)
            loss, mse_loss = model.ssl_train_forward(eda,bvp,temp, mask_ratio=mask_ratio, num_masked=num_masked)
            print('MSE_Loss',mse_loss)
            epoch_mse_loss += mse_loss
          
            
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if train_mode in ['supervised', 'frozen', 'fine-tuned']:
            _, predict_targets = torch.max(pred.detach().data, 1)
            epoch_acc += (predict_targets.cpu() == y.detach().cpu()).sum().float()/y.size(0)
        
            train_loader.set_postfix(loss=epoch_loss/batch_idx, accuracy=epoch_acc/batch_idx, time = time.time()-start)
        
        if train_mode =='ssl':
            train_loader.set_postfix(mse_loss=epoch_mse_loss/batch_idx, time = time.time()-start)
        
        
    return epoch_loss, epoch_acc

def validate(model, val_loader, device):
    model.eval()
    val_acc = 0
    total = 0
    print('*****************validating*******************')
    for batch_idx, data in enumerate(val_loader, start=1):
        
        x = data[0].float().to(device)
        y = data[1].long().to(device)
        eda=x[:, :, 0]
        #print(eda.shape)
        bvp=x[:, :, 1]
            #bvp = sample_batched["bvp"].type(torch.float32)
            #temp = sample_batched["temp"].type(torch.float32)
        temp=x[:, :, 2]
        
        outputs, _ = model(eda,bvp,temp)
        _, predict_targets = torch.max(outputs.detach().data, 1)
        true_targets = y.detach().cpu()
        val_acc += (predict_targets.cpu() == true_targets).sum().float()
        total += true_targets.size(0)
    
    val_acc /= total
    print('Validation acc: %.2f' % (val_acc*100))
    return val_acc

