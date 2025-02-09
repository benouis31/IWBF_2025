import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
import numpy as np

import logging
import pathlib
import numpy as np
import h5py
import torch
import random
from torch.utils.data.sampler import SubsetRandomSampler
from collections import Counter
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from sklearn.preprocessing import LabelEncoder
from load_data import  WESADDataset_Ba,VERBIODataset_train_valid_Ba, create_balanced_dataloader







verbio_subjects = ['P001', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008',
                   'P009','P011', 'P012', 'P013', 'P014', 'P016', 'P017',
                   'P018', 'P020','P021', 'P023', 'P026', 'P027', 'P031',
                   'P032', 'P035', 'P037','P038', 'P039', 'P040', 'P041',
                   'P042', 'P043', 'P044', 'P045','P046', 'P047', 'P048',
                   'P049', 'P050', 'P051', 'P052', 'P053','P056', 'P057',
                   'P058', 'P060', 'P061', 'P062', 'P063', 'P064','P065',
                   'P066', 'P067', 'P068', 'P071', 'P072', 'P073']



selected_subjects = ['S1','S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']


path_2= '/VerBIO-norm.hdf5'
path_1= '/WESAD-norm.hdf5'


   




def data_generator(path_1,path_2, configs, training_mode):
    # VerBIO path_2
    # WESAD  path_1
    
   
   
  
    
    train_dataset = WESADDataset_Ba(path_1, configs,selected_subjects, mode="train")
    val_dataset=WESADDataset_Ba(path_1, configs,selected_subjects, mode="valid")
    
    test_dataset = WESADDataset_Ba(path_1, configs,selected_subjects, mode="test",
                               modality_masking=False, modality_mask_percentage=0.3,
                               label_drop=False, label_drop_percentage=0.2)
                               
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=True,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
   

    return train_loader, valid_loader, test_loader






