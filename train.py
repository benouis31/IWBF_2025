import torch

import os
import numpy as np
from datetime import datetime
import argparse


from loader import data_generator


from model import TMS_BIO

import utils

parser = argparse.ArgumentParser()

#SEEDS = [25, 45, 65, 85, 105,56,10,55,90]
SEEDS = [25]


if __name__ == '__main__':
    device = 'cpu:1'
    train_mode = 'ssl'
    data_type = 'WESAD'
    data_path = f"./IWBF_2025/data/{data_type}"

    config_module = __import__(f'config_files.{data_type}_Configs', fromlist=['Config'])
    configs = config_module.Config()
    
    for SEED in SEEDS:
        ckpt = f'./IWBF_2025/checkpoints/{data_type}/{train_mode}_{SEED}.pth'
    
        os.makedirs(f'/IWBF_2025/checkpoints/{data_type}/', exist_ok=True)
    
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print(configs)
        path_1= 'IWBF_2025/data/VerBIO-norm.hdf5'
        path_2= '/IWBF_2025/data/WESAD-norm.hdf5'
        
        train_dl, valid_dl, test_dl = data_generator(path_1,path_2, configs, train_mode)
        
        model = TMS_BIO(configs)
      
        
        if train_mode in ['fine-tuned', 'Frozen']:
            model.load_state_dict(torch.load(f'/IWBF_2025/checkpoints/{data_type}/ssl_{SEED}.pth'))
        
        model = model.to(device)
        
        if train_mode == 'Frozen':
            optimizer=torch.optim.AdamW(model.linear.parameters(), lr=configs.lr, weight_decay=0.05)
        elif train_mode == 'Fine-tuned':
            optimizer = torch.optim.AdamW([
                {"params":model.conv_block.parameters(), "lr_mult": 0.1},
                {"params":model.transformer_encoder.parameters(), "lr_mult": 0.1},
                {"params":model.linear.parameters(), "lr_mult": 1.0}],
                lr=configs.lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=0.05)
            
        best_acc = 0.
        #train_dl, valid_dl, test_dl = data_generator(data_path, configs, train_mode)
        
        for epoch in range(1, configs.num_epoch + 1):
            
            print(f'Epoch: {epoch}|{configs.num_epoch} || Seed: {SEED}')
            epoch_loss, epoch_acc = utils.train_epoch(model, train_dl, optimizer, train_mode, device)
            if train_mode in ['supervised', 'Frozen', 'fine-tuned']:
                val_acc = utils.validate(model, valid_dl, device)
                if best_acc < val_acc:
                    print('Save best acc')
                    best_acc = val_acc
                    torch.save(model.state_dict(), ckpt)
            else:
                torch.save(model.state_dict(), ckpt)
