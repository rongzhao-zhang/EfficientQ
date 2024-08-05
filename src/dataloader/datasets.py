#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rongzhao Zhang
"""

import os.path as P
import numpy as np
import torch
import pickle

__all__ = ['Dataset_SEG', 'Dataset_SEG_OnDisk' ]


#%% Data access methods

def access_npy(data_dir, mod, sn, dtype):
    fname = P.join(data_dir, mod, '%s.npy' % sn)
    data = np.load(fname)
    if data.dtype != dtype:
        data = data.astype(dtype)
    return data

def access_npz(data_dir, mod, sn, dtype):
    fname = P.join(data_dir, mod, '%s.npz' % sn)
    data = np.load(fname, allow_pickle=True)['arr_0']
    if data.dtype != dtype:
        data = data.astype(dtype)
    return data

def access_memmap(data_dir, mod, sn, dtype, shapes):
    fname = P.join(data_dir, mod, '%s.dat' % sn)
    data = np.memmap(fname, dtype=dtype, mode='r', shape=shapes[sn])
    return data

ACCESS_MAP = {'npy': access_npy, 'npz': access_npz, 'memmap': access_memmap}

#%%
class Dataset_SEG(torch.utils.data.Dataset):
    """
    General Dataset for Segmentation
    INPUT:
        DATA_DIR   -- Data directory
        SPLIT      -- Path of the split file (sn)
                      e.g. 'splits/train_128.txt'
        MODALITIES -- Tuple of modality names, label must be the first one
                      e.g. ('label', 'DWI', 'ADC')
        ACCESS_TYPE-- Data access method {'npy', 'npz', 'memmap'}
        TRANSFORM  -- Transform applied to input dataloader (default = None)
                      e.g. transform=my.transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(),])

    OUTPUT ( RETURNED BY __GETITEM__ ) :
        (IMG, LABEL) -- (C, [D,] H, W) as np.float32 and ([D,] H, W) as np.uint8 for ''img'' and ''label'' respectively
    """
    def __init__(self, data_dir, split, modalities, access_type='npz', transform_rand=None,
                 transform_fix=None):
        # define two transforms so that the dataset can switch between trainloader and trainseqloader
        self.transform_rand = transform_rand
        self.transform_fix = transform_fix
        if transform_rand:
            self.transform = transform_rand
        else:
            self.transform = transform_fix

        self.data = []
        self.label = []
        
        # Load subject names into a list
        sn_list = open(split, 'r').read().splitlines()
        sn_list.sort()
        
        access_data = ACCESS_MAP[access_type]
        if access_type == 'memmap':
            with open(P.join(data_dir, 'shapes.pickle'), 'rb') as F:
                shapes = pickle.load(F)
            access_data = lambda *p: access_memmap(*p, shapes)
        for sn in sn_list:
            img_ = []
            for mod in modalities[1:]:
                image = access_data(data_dir, mod, sn, 'float32')
                img_.append(image)
            img_ = np.stack(img_)
            self.data.append(img_)
            
            if modalities[0] is not None:
                label_ = access_data(data_dir, modalities[0], sn, 'uint8')
                self.label.append(label_)
            else: # no label is provided, using image as a placeholder
                self.label.append(image.astype('uint8'))
            
    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label
        
    def __len__(self):
        return len(self.data)
    
    def use_random_transform(self):
        self.transform = self.transform_rand
        if self.transform is None:
            raise RuntimeWarning('transform_rand is None.')
    
    def use_fix_transform(self):
        self.transform = self.transform_fix
        if self.transform is None:
            raise RuntimeWarning('transform_fix is None.')


class Dataset_SEG_OnDisk(torch.utils.data.Dataset):
    """
    General Dataset for Segmentation
    INPUT:
        DATA_DIR   -- Data directory
        SPLIT      -- Path of the split file (sn)
                      e.g. 'splits/train_128.txt'
        MODALITIES -- Tuple of modality names, label must be the first one
                      e.g. ('label', 'DWI', 'ADC')
        ACCESS_TYPE-- Data access method {'npy', 'npz', 'memmap'}
        TRANSFORM  -- Transform applied to input dataloader (default = None)
                      e.g. transform=my.transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(),])

    OUTPUT ( RETURNED BY __GETITEM__ ) :
        (IMG, LABEL) -- (C, [D,] H, W) as np.float32 and ([D,] H, W) as np.uint8 for ''img'' and ''label'' respectively
    """
    def __init__(self, data_dir, split, modalities, access_type='npz', transform_rand=None,
                 transform_fix=None):
        self.data_dir = data_dir
        self.modalities = modalities
        self.transform_rand = transform_rand
        self.transform_fix = transform_fix
        if transform_rand:
            self.transform = transform_rand
        else:
            self.transform = transform_fix
        
        # Load subject names into a list
        self.sn_list = open(split, 'r').read().splitlines()
        
        self.access_data = ACCESS_MAP[access_type]
        if access_type == 'memmap':
            with open(P.join(data_dir, 'shapes.pickle'), 'rb') as F:
                shapes = pickle.load(F)
            self.access_data = lambda *p: access_memmap(*p, shapes)
        
    def __getitem__(self, index):
        # load image
        img_ = []
        for mod in self.modalities[1:]:
            image = self.access_data(self.data_dir, mod, self.sn_list[index], 'float32')
            img_.append(image)
        img_ = np.stack(img_)
        # load label
        if self.modalities[0] is not None:
            label_ = self.access_data(self.data_dir, self.modalities[0], self.sn_list[index],
                                      'uint8')
        else:  # no label is provided, using image as a placeholder
            label_ = image.astype('uint8')
        
        if self.transform is not None:
            img, label = self.transform(img_, label_)
#            print(img.shape)

        return img, label
        
    def __len__(self):
        return len(self.sn_list)
    
    def use_random_transform(self):
        self.transform = self.transform_rand
        if self.transform is None:
            raise RuntimeWarning('transform_rand is None.')
    
    def use_fix_transform(self):
        self.transform = self.transform_fix
        if self.transform is None:
            raise RuntimeWarning('transform_fix is None.')

