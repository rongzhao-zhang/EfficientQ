#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""
import os.path as P
from torch.utils.data import DataLoader
from . import transforms, datasets


# copied from utils.misc
def file_to_dict(fname, sep=','):
    if fname is None:
        return None
    with open(fname, 'r') as f:
        lines = f.read().splitlines()
    d = dict()
    for line in lines:
        k, v = line.split(sep)
        d[k] = v
    return d


class DataHub_SEG(object):
    def __init__(self, data_dir, modalities, train_split=None, val_split=None, 
                 test_split=None, true_test_split=None, train_batchsize=32,
                 test_batchsize=32, std=1, mean=0, access_type='npz', 
                 rand_flip=None, crop_type=None, crop_size_img=None, crop_size_label=None, 
                 balance_rate=0.5, balance_mask_func=None, train_pad_size=None, test_pad_size=None, mod_drop_rate=0, 
                 train_drop_last=False, crop_type_test=None, crop_size_img_test=None, 
                 crop_size_label_test=None, DataSet=datasets.Dataset_SEG,
                 label_loader_path=None, rand_rot90=False, random_noise_prob=None,
                 num_workers=1, random_black_patch_size=None, 
                 mini_positive=None, sn_fn_file=None, scale_bound=None, scale_order=1, scale_p=0.5,
                 slide_patch_size=None, slide_overlap=None, device='cuda', tfm_lambda=None):
        self.data_dir = data_dir
        self.std = std
        self.mean = mean
        self.access_type = access_type
        self.num_workers = num_workers
        self.sn_fn_file = sn_fn_file
        self.slide_patch_size = slide_patch_size
        self.slide_overlap = slide_overlap
        self.device = device

        # attributes to define and deliver
        self.train_sn = self.val_sn = self.test_sn = self.true_test_sn \
            = self.trainloader = self.trainseqloader = self.valloader = self.testloader \
            = self.true_test_image_loader = None
        # subject name to file name mapping
        self.sn_to_fn_map = file_to_dict(P.join(data_dir, sn_fn_file))
        
        
        if train_split and P.isfile(train_split):
            with open(train_split, 'r') as f:
                self.train_sn = f.read().splitlines()
        if val_split and P.isfile(val_split):
            with open(val_split, 'r') as f:
                self.val_sn = f.read().splitlines()
        if test_split and P.isfile(test_split):
            with open(test_split, 'r') as f:
                self.test_sn = f.read().splitlines()
        if true_test_split and P.isfile(true_test_split):
            with open(true_test_split, 'r') as f:
                self.true_test_sn = f.read().splitlines()
                
        if P.exists(P.join(data_dir, 'meanstd.txt')):
            with open(P.join(data_dir, 'meanstd.txt'), 'r') as f:
                lines = f.read().splitlines()
            self.mean = [ float(x) for x in lines[0].split()[1:] ]
            self.std = [ float(x) for x in lines[1].split()[1:] ]
            print('import mean and std value from file \'meanstd.txt\'')
            print('mean = %s, std = %s' % (str(self.mean), str(self.std)))

        self.basic_transform_ops = [transforms.ToTensor(), 
                                    transforms.Normalize(self.mean, self.std)]

        train_transform = \
        self._make_train_transform(crop_type, crop_size_img, crop_size_label,
                                   rand_flip, mod_drop_rate, balance_rate, balance_mask_func,
                                   train_pad_size, rand_rot90, random_black_patch_size,
                                   mini_positive, scale_bound, scale_order, scale_p, random_noise_prob)
        test_transform = \
        self._make_test_transform(crop_type_test, crop_size_img_test,
                                  crop_size_label_test, test_pad_size)
        
        if tfm_lambda:
            train_transform.transforms.append(transforms.Lambda(tfm_lambda))
            test_transform.transforms.append(transforms.Lambda(tfm_lambda))
        
        if self.train_sn:
            train_dataset = DataSet(data_dir, train_split, modalities, access_type, 
                                    transform_rand=train_transform, transform_fix=test_transform)
            self.trainloader = DataLoader(train_dataset, train_batchsize, shuffle=True, 
                                          num_workers=num_workers, drop_last=train_drop_last)
            self.trainseqloader = DataLoader(train_dataset, test_batchsize, shuffle=False, 
                                             num_workers=num_workers, drop_last=False)
        if self.val_sn:
            val_dataset = DataSet(data_dir, val_split, modalities, access_type, 
                                  transform_rand=None, transform_fix=test_transform)
            self.valloader = DataLoader(val_dataset, test_batchsize, shuffle=False, 
                                        num_workers=num_workers, drop_last=False)
        if self.test_sn:
            test_dataset = DataSet(data_dir, test_split, modalities, access_type, 
                                   transform_rand=None, transform_fix=test_transform)
            self.testloader = DataLoader(test_dataset, test_batchsize, shuffle=False, 
                                         num_workers=num_workers, drop_last=False)
        if self.true_test_sn:
            modal_t = list(modalities).copy()
            modal_t[0] = None
            true_test_data_set = DataSet(data_dir, true_test_split, modal_t, access_type, 
                                         transform_fix=test_transform)
            self.true_test_image_loader = \
                            DataLoader(true_test_data_set, batch_size=test_batchsize, shuffle=False,
                                       num_workers=num_workers, drop_last=False)
        
    def _make_train_transform(self, crop_type, crop_size_img, crop_size_label,
                             rand_flip, mod_drop_rate, balance_rate, balance_mask_func, pad_size,
                             rand_rot90, random_black_patch_size, mini_positive,
                             scale_bound, scale_order, scale_p, random_noise_prob):
        train_transform_ops = self.basic_transform_ops.copy()
            
        if random_black_patch_size is not None:
            train_transform_ops.append(transforms.RandomBlack(random_black_patch_size))
        if mod_drop_rate > 0:
            train_transform_ops.append(transforms.RandomDropout(mod_drop_rate))
        if rand_flip is not None:
            train_transform_ops.append(transforms.RandomFlip(rand_flip))
        if pad_size is not None:
            train_transform_ops.append(transforms.Pad(pad_size, 0))
        if rand_rot90:
            train_transform_ops.append(transforms.RandomRotate2d())

        if crop_type == 'random':
            if mini_positive:
                train_transform_ops.append(transforms.RandomCropMinSize(crop_size_img, mini_positive))
            elif scale_bound:
                train_transform_ops.append(transforms.RandomScaleCrop(scale_bound[0], 
                                                                      scale_bound[1], 
                                                                      crop_size_img,
                                                                      scale_order, scale_p))
            else:
                train_transform_ops.append(transforms.RandomCrop(crop_size_img))
        elif crop_type == 'balance':
            train_transform_ops.append(transforms.BalanceCrop(balance_rate, crop_size_img,
                                                              crop_size_label, balance_mask_func))
        elif crop_type == 'center':
            train_transform_ops.append(transforms.CenterCrop(crop_size_img,
                                                          crop_size_label))
        elif crop_type is None:
            pass
        else:
            raise RuntimeError('Unknown train crop type.')
        
        if random_noise_prob:
            train_transform_ops.append(transforms.RandomNoise(random_noise_prob, max_scale=0.3))

        return transforms.Compose(train_transform_ops)

    def _make_test_transform(self, crop_type, crop_size_img, crop_size_label, pad_size):
        test_transform_ops = self.basic_transform_ops.copy()
        if pad_size is not None:
            test_transform_ops.append(transforms.Pad(pad_size, 0))
        if crop_type == 'center':
            test_transform_ops.append(transforms.CenterCrop(crop_size_img,
                                                          crop_size_label))
        elif crop_type is None:
            pass
        else:
            raise RuntimeError('Unknown test crop type.')

        return transforms.Compose(test_transform_ops)

