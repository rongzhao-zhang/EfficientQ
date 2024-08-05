#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:11:54 2017

@author: zrz
"""

from __future__ import division
import torch
import random
import numbers
import types
import collections
import numpy as np
from scipy import ndimage

__all__ = ['Compose', 'ToTensor', 'Normalize', 'Lambda', 'CenterCrop', 'RandomCrop',
           'RandomTransverseFlip', 'RandomSagittalFlip', 'RandomVerticalFlip',
           'RandomFlip', 'RandomDropout', 'RandomBlack', 'RandomScaleCrop']

def passthrough(img, label):
    return img, label

def has_even(intseq):
    for i in intseq:
        if i % 2 == 0:
            return True
    return False

def crop_size_correct(sp, ep, this_size):
    assert ep-sp <= this_size, 'Invalid crop size: ep = %d, sp = %d, this_size = %d' % (ep, sp, this_size)
    if sp < 0:
        ep -= sp
        sp -= sp
    elif ep > this_size:
        sp -= (ep-this_size)
        ep -= (ep-this_size)

    return sp, ep

def crop(tensor, locations):
    """ Crop on the inner-most 2 or 3 dimensions
    ''location'' is a tuple indicating locations of start and end points
    """
#    locations = [int(x) for x in locations]
    s = tensor.size()
    if len(locations) == 6:
        x1, y1, z1, x2, y2, z2 = locations
        x1, x2 = crop_size_correct(x1, x2, s[-3])
        y1, y2 = crop_size_correct(y1, y2, s[-2])
        z1, z2 = crop_size_correct(z1, z2, s[-1])
        return tensor[..., x1:x2, y1:y2, z1:z2]
    elif len(locations) == 4:
        x1, y1, x2, y2 = locations
        x1, x2 = crop_size_correct(x1, x2, s[-2])
        y1, y2 = crop_size_correct(y1, y2, s[-1])
        return tensor[..., x1:x2, y1:y2]
    else:
        raise RuntimeError('Invalid crop size dimension.')

def center_crop(tensor, size):
    if len(size) == 3:
        d, h, w = tensor.size()[-3:]
        td, th, tw = size
        if d == td and w == tw and h == th:
            return tensor

        z1 = (w - tw) // 2
        y1 = (h - th) // 2
        x1 = (d - td) // 2
        loc = (x1, y1, z1, x1 + td, y1 + th, z1 + tw)
        return crop(tensor, loc)
    elif len(size) == 2:
        h, w = tensor.size()[-2:]
        th, tw = size
        if w == tw and h == th:
            return tensor

        y1 = (w - tw) // 2
        x1 = (h - th) // 2
        loc = (x1, y1, x1 + th, y1 + tw)
        return crop(tensor, loc)
    else:
        raise RuntimeError('Invalid center crop size.')

def crop_centroid(tensor, centroid, size):
    """ Crop on the inner-most 2 or 3 dimensions
    ''centroid'' is a tuple indicating locations of the centroid
    ''size'' is a tuple indicating the size of the cropped patch
    """
    assert len(centroid) == len(size), 'Mismatched centroid and size: %s, %s' % (str(centroid), str(size))
    s = [int(ss) // 2 for ss in size]
    start_pos = [ci-si for ci, si in zip(centroid, s)]
    end_pos = [start_pos_i + size_i for start_pos_i, size_i in zip(start_pos, size)]
    if len(centroid) == 3:
        locations = (start_pos[0], start_pos[1], start_pos[2], end_pos[0], end_pos[1], end_pos[2])
        return crop(tensor, locations)
    elif len(centroid == 2):
        locations = (start_pos[0], start_pos[1], end_pos[0], end_pos[1])
        return crop(tensor, locations)
    else:
        raise RuntimeError('Invalid centroid crop size.')


def flip_tensor(tensor, axis):
    if len(tensor.size()) == 1:
        return tensor
    tNp = np.flip(tensor.numpy(), axis).copy()
    return torch.from_numpy(tNp)


def rot90_tensor(tensor, k=1):
    if len(tensor.size()) == 2:
        tNp = np.rot90(tensor.numpy(), k).copy()
    elif len(tensor.size()) == 3:
        tNp = np.rot90(tensor.numpy(), k, (1,2)).copy()
    else:
        tNp = tensor.numpy()
    return torch.from_numpy(tNp)


class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class ToTensor(object):
    """ Convert np.ndarray to torch.*Tensor """
    def __call__(self, img, label):
        return torch.from_numpy(img.copy()).float(), torch.from_numpy(label.copy()).long()
    
class LabelBinary(object):
    def __call__(self, tensor, label):
        return tensor, (label>0).long()


class Normalize(object):
    """ Normalize ''tensor'' by ''mean'' and ''std'' along each channel if corresponding arguments are provided.
        Other normalize to zero mean and unit std by channel.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, label):
        if self.mean is None:
            return tensor, label

        tensor = tensor.clone()
        if isinstance(self.mean, collections.Iterable):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        else:
            t.sub_(self.mean).div_(self.std)
        return tensor, label
    
class ToDevice(object):
    '''Move Tensor to the specified device'''
    def __init__(self, device='cuda:0'):
        self.device = device

    def __call__(self, tensor, label):
        if tensor.device != torch.device(self.device) or label.device != torch.device(self.device):
            return tensor.to(self.device), label.to(self.device)
        else:
            return tensor, label

class Scale(object):
    """ TO BE IMPLEMENTED
    Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self):
        pass

    def __call__(self, img, label):
        return img, label

class Pad(object):
    """ Pad the input image to specified size with default 'pad_value' 0. """
    def __init__(self, size, pad_value=0):
        if len(size) == 3:
            self.padder = Pad3d(size, pad_value)
        elif len(size) == 2:
            self.padder = Pad2d(size, pad_value)
        else:
            raise RuntimeError('Invalid center crop size.')

    def __call__(self, img, label):
        img_new, label_new = self.padder(img, label)
        return img_new, label_new

class Pad3d(object):
    """ Pad the 3D input image to specified size with default 'pad_value' 0. """
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img, label):
        new_size = [0, 0, 0]
        # If padding size is smaller than original size, keep the original size
        c = 0
        for i in range(3):
            if img.size(i+1) < self.size[i]:
                new_size[i] = self.size[i]
            else:
                new_size[i] = img.size(i+1)
                c += 1
        if c == 3:
            return img, label
        img_new = (torch.ones(img.size(0), *new_size) * self.pad_value).float()
        label_new = torch.zeros(*new_size).long()
        new_size_np = np.array(new_size)
        old_size_np = np.array(label.size())
        start_pos = (new_size_np - old_size_np) // 2
        end_pos = start_pos + old_size_np
#        print('new size is', new_size)
#        print('start_pos is', start_pos)
#        print('end_pos is', end_pos)
#        print('img_new size is', img_new.size())
#        print('img size is', img.size())
        img_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1],
                start_pos[2]:end_pos[2]] = img
        label_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1],
                start_pos[2]:end_pos[2]] = label
        return img_new, label_new

class Pad2d(object):
    """ Pad the 2D input image to specified size with default 'pad_value' 0. """
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img, label):
        new_size = [0, 0]
        # If padding size smaller than original size, keep the original size
        c = 0
        for i in range(2):
            if img.size(i+1) <= self.size[i]:
                new_size[i] = self.size[i]
            else:
                new_size[i] = img.size(i+1)
                c += 1
        if c == 2:
            return img, label
        img_new = (torch.ones(img.size(0), *new_size) * self.pad_value).float()
        label_new = torch.zeros(*new_size).long()
        new_size_np = np.array(new_size)
        old_size_np = np.array(label.size())
        start_pos = (new_size_np - old_size_np) // 2
        end_pos = start_pos + old_size_np
        img_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1]] = img
        label_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1]] = label
        return img_new, label_new

class Lambda(object):
    """Applies a lambda as a transform"""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, img, label):
        return self.lambd(img, label)

class CenterCrop(object):
    def __init__(self, size, size_label):
        if len(size) == 3:
            self.cropper = CenterCrop3d(size, size_label)
        elif len(size) == 2:
            self.cropper = CenterCrop2d(size, size_label)
        else:
            raise RuntimeError('Invalid center crop size.')

    def __call__(self, img, label):
        return self.cropper(img, label)

class CenterCrop3d(object):
    def __init__(self, size, size_label=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size),) * 3
        else:
            self.size = size
        if size_label is None:
            self.size_label = self.size
        elif isinstance(size_label, numbers.Number):
            self.size_label = (int(size_label),) * 3
        else:
            self.size_label = size_label

    def __call__(self, img, label):
        return center_crop(img, self.size), \
               center_crop(label, self.size_label)

class CenterCrop2d(object):
    def __init__(self, size, size_label=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size),) * 2
        else:
            self.size = size
        if size_label is None:
            self.size_label = self.size
        elif isinstance(size_label, numbers.Number):
            self.size_label = (int(size_label),) * 2
        else:
            self.size_label = size_label

    def __call__(self, img, label):
        return center_crop(img, self.size), \
               center_crop(label, self.size_label)

class RandomCrop(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size MUST be a 3-tuple (target_depth, target_height, target_width)
    or a 2-tuple (target_height, target_width). The dimensionality is automatically
    detected according to the length of the size tuple.
    """
    def __init__(self, size):
        if len(size) == 3:
            self.cropper = RandomCrop3d(size)
        elif len(size) == 2:
            self.cropper = RandomCrop2d(size)
        else:
            raise RuntimeError('Invalid random crop size.')

    def __call__(self, img, label):
        return self.cropper(img, label)

class RandomCropMinSize(object):
    """Crops the given (img, label) at a random location to be of the given size, 
    while ensuring the number of positive pixels is either zero or larger than 
    a minimal value. Size MUST be a 3-tuple (target_depth, target_height, target_width)
    or a 2-tuple (target_height, target_width). The dimensionality is automatically
    detected according to the length of the size tuple.
    """
    def __init__(self, size, mini_positive=0):
        self.mini_positive = mini_positive
        if len(size) == 3:
            self.cropper = RandomCrop3d(size)
        elif len(size) == 2:
            self.cropper = RandomCrop2d(size)
        else:
            raise RuntimeError('Invalid random crop size.')

    def __call__(self, img, label):
        imgc, labelc = self.cropper(img, label)
        count = 0
        while(0 < labelc.sum() < self.mini_positive):
            imgc, labelc = self.cropper(img, label)
            count += 1
        if count > 0:
            print('Crop %d times for a valid positive size.' % count)
        return imgc, labelc

class RandomCrop3d(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size can be a tuple (target_depth, target_height, target_width)
    or an integer, in which case the target will be of a cubic shape (size, size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        d, h, w = img.size()[-3:]
        td, th, tw = self.size
        assert td<=d and th<=h and tw<=w, 'td=%d,d=%d; th=%d,h=%d; tw=%d,w=%d' % (td,d,th,h,tw,w)
        if d == td and w == tw and h == th:
            return img, label
#        print('d:', d, 'h:', h, 'w:', w)
#        print('td:', td, 'th:', th, 'tw:', tw)
        z1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        x1 = random.randint(0, d - td)
        loc = (x1, y1, z1, x1 + td, y1 + th, z1 + tw)
        return crop(img, loc), crop(label, loc)

class RandomCrop2d(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size can be a tuple (target_depth, target_height, target_width)
    or an integer, in which case the target will be of a cubic shape (size, size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        h, w = img.size()[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return img, label

        y1 = random.randint(0, w - tw)
        x1 = random.randint(0, h - th)
        loc = (x1, y1, x1 + th, y1 + tw)
        return crop(img, loc), crop(label, loc)

class BalanceCrop(object):
    """Randomly crop the given image and label with balanced centroid class.
    """
    def __init__(self, positive_prob, img_size, label_size=None, mask_func=None):
        self.prob = positive_prob
        if label_size is None:
            label_size = img_size
        if isinstance(img_size, numbers.Number):
            img_size = (int(img_size),) * 3
        if isinstance(label_size, numbers.Number):
            label_size = (int(label_size),) * 3

        self.img_size = img_size
        self.label_size = label_size
#        self.mask = None
        if mask_func is None:
            self.mask_func = lambda label: label>0
        else:
            self.mask_func = mask_func

    def __call__(self, img, label):
        # Note: img is CxDxHxW, label is DxHxW
#        if self.mask is None or self.mask.size() != img.size():
#            self._make_mask(label.size())
        mask = self.mask_func(label)
        positive_loc = torch.nonzero(mask, as_tuple=False) #*self.mask
#        print('mask shape', mask.shape)
#        print('positive_loc shape', positive_loc.shape)
        negative_loc = torch.nonzero(mask==0, as_tuple=False) #*self.mask

        if len(negative_loc) == 0 and len(positive_loc) == 0:
            raise RuntimeError('Invalid patch size.')
        elif len(negative_loc) == 0:
            is_positive = True
        elif len(positive_loc) == 0:
            is_positive = False
        else:
            is_positive = random.random() <= self.prob

        if is_positive:
            i = random.randint(0, len(positive_loc)-1)
            center_loc = positive_loc[i]
        else:
            i = random.randint(0, len(negative_loc)-1)
            center_loc = negative_loc[i]
#        print(img.shape, self.img_size)
#        for i in range(len(img[0].shape)):
#            assert img[0].shape[i] >= self.img_size[i], '%s vs %s' % (img[0].shape, self.img_size)
        return crop_centroid(img, center_loc, self.img_size), \
               crop_centroid(label, center_loc, self.label_size)


class RandomTransverseFlip(object):
    """ Randomly transverse flips the given Tensor along 'w' dimension
    """
    def __call__(self, img, label):
        return flip_tensor(img, -1), flip_tensor(label, -1)

class RandomSagittalFlip(object):
    """ Randomly sagittally flips the given Tensor along 'h' dimension
    """
    def __call__(self, img, label):
        return flip_tensor(img, -2), flip_tensor(label, -2)

class RandomVerticalFlip(object):
    """ Randomly vertically flips the given Tensor along 'd' dimension
    """
    def __call__(self, img, label):
        return flip_tensor(img, -3), flip_tensor(label, -3)

class RandomFlip(object):
    def __init__(self, axis_switch=None):
        if len(axis_switch) == 3:
            self.flipper = RandomFlip3d(axis_switch)
        elif len(axis_switch) == 2:
            self.flipper = RandomFlip2d(axis_switch)
        elif axis_switch == None:
            self.flipper = passthrough
        else:
            raise RuntimeError('Invalid random flip controller.')

    def __call__(self, img, label):
        return self.flipper(img, label)

class RandomFlip3d(object):
    def __init__(self, axis_switch=(1,1,1)):
        self.axis_switch = axis_switch

    def __call__(self, img, label):
        if self.axis_switch[0]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -3)
                label = flip_tensor(label, -3)
        if self.axis_switch[1]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -2)
                label = flip_tensor(label, -2)
        if self.axis_switch[2]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -1)
                label = flip_tensor(label, -1)
        return img, label

class RandomFlip2d(object):
    def __init__(self, axis_switch=(1,1)):
        self.axis_switch = axis_switch

    def __call__(self, img, label):
        if self.axis_switch[0]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -2)
                label = flip_tensor(label, -2)
        if self.axis_switch[1]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -1)
                label = flip_tensor(label, -1)
        return img, label


class RandomScaleCrop(object):
    """ Random crop on randomly scaled images.
    """
    def __init__(self, l_scale, h_scale, size, scale_order, p=0.5):
        self.l_scale = l_scale
        self.h_scale = h_scale
        self.size = size
        self.p = p
        self.crop_only = RandomCrop(size)
        if len(size) == 3:
            self.doer = RandomScaleCrop3d(l_scale, h_scale, size, scale_order)
        elif len(size) == 2:
            self.doer = RandomScaleCrop2d(l_scale, h_scale, size, scale_order)
        else:
            raise RuntimeError('Invalid size for RandomScaleCrop.')

    def __call__(self, img, label):
        if random.random() < self.p:
            return self.doer(img, label)
        else:
            return self.crop_only(img, label)

class RandomScaleCrop3d(object):
    """ Perform random crop on randomly scaled 3D images.
    """
    def __init__(self, l_scale, h_scale, size, scale_order):
        self.l_scale = l_scale
        self.h_scale = h_scale
        self.size = size
        self.scale_order = scale_order

    def __call__(self, img, label):
        # random crop size
        crop_size = np.array(self.size)
        d, h, w = img.size()[-3:]
        # random scale factor
        fmin = (crop_size[0]/d, crop_size[1]/h, crop_size[2]/w)
        if isinstance(self.l_scale, collections.Iterable):
            l_scale = [max(x, y) for x, y in zip(fmin, self.l_scale)]
            h_scale = self.h_scale
            factor = np.random.uniform(l_scale, h_scale, (3,))
        else:
            factor = (np.random.uniform(max(self.l_scale, max(fmin)), self.h_scale),) * 3
        # crop the receptive field before rescale, ceiling to ensure enough space
        td, th, tw = [int(np.ceil(x/y)) for x, y in zip(crop_size, factor)]
        
#        print('factor:', factor, 'fmin:', fmin)
#        print('d:', d, 'h:', h, 'w:', w)
#        print('td:', td, 'th:', th, 'tw:', tw)
        
        x1 = random.randint(0, d - td)
        y1 = random.randint(0, h - th)
        z1 = random.randint(0, w - tw)
        loc = (x1, y1, z1, x1 + td, y1 + th, z1 + tw)
        
#        print('factor', factor)
#        print('Img size:', img.size())
#        print('x1:', x1, 'y1:', y1, 'z1:', z1)
        img_patch, label_patch = crop(img, loc), crop(label, loc)
        img_pat_np, label_pat_np = img_patch.numpy(), label_patch.numpy()
        # rescale the cropped patch
        channel_stack = []
        for i in range(len(img_pat_np)):
            channel_stack.append(ndimage.zoom(img_pat_np[i], factor, order=self.scale_order)) # resample
        img_pat_np = np.stack(channel_stack)
        
        pmax, pmin = label_pat_np.max(), label_pat_np.min()
        label_pat_np = ndimage.zoom(label_pat_np, factor, order=0) # resample
        # eliminate illegal values: not necessary for bilinear resampling
        if self.scale_order >= 2:
            bigger = label_pat_np>pmax
            smaller = label_pat_np<pmin
            if bigger.any() or smaller.any():
#                print('bigger', bigger.sum())
#                print('smaller', smaller.sum())
                label_pat_np[bigger] = pmax
                label_pat_np[smaller] = pmin
        
        img_patch = torch.from_numpy(img_pat_np).float()
        label_patch = torch.from_numpy(label_pat_np).long()
        
#        print('second crop', img_patch.size())
        img_patch, label_patch = crop(img_patch, (0,0,0,*crop_size)), crop(label_patch, (0,0,0,*crop_size))
        
#        print(img_patch.dtype, img_patch.max(), img_patch.min(), img_patch.size())
#        print(label_patch.dtype, label_patch.max(), label_patch.min(), label_patch.size())
        
        # crop again to eliminate extra pixels
        return img_patch, label_patch
    
class RandomScaleCrop2d(object):
    """ Random crop on randomly scaled 3D images.
    """
    def __init__(self, l_scale, h_scale, size, scale_order):
        self.l_scale = l_scale
        self.h_scale = h_scale
        self.size = size

    def __call__(self, img, label):
        # random crop size
        crop_size = np.array(self.size)
        h, w = img.size()[-2:]
        # random scale factor
        fmin = (crop_size[0]/h, crop_size[1]/w)
        if isinstance(self.l_scale, collections.Iterable):
            l_scale = [max(x, y) for x, y in zip(fmin, self.l_scale)]
            h_scale = self.h_scale
            factor = np.random.uniform(l_scale, h_scale, (2,))
        else:
            factor = (np.random.uniform(max(self.l_scale, max(fmin)), self.h_scale),) * 2
        # crop the receptive field before rescale, ceiling to ensure enough space
        tw, th, td = [int(np.ceil(x/y)) for x, y in zip(crop_size, factor)]
        
        y1 = random.randint(0, w - tw)
        x1 = random.randint(0, h - th)
        loc = (x1, y1, x1 + th, y1 + tw)
        img_patch, label_patch = crop(img, loc), crop(label, loc)
        img_pat_np, label_pat_np = img_patch.numpy(), label_patch.numpy()
        # rescale the cropped patch
        channel_stack = []
        for i in range(len(img_pat_np)):
            channel_stack.append(ndimage.zoom(img_pat_np[i], factor, order=self.scale_order)) # bilinear resample
        img_pat_np = np.stack(channel_stack)
            
        pmax, pmin = label_pat_np.max(), label_pat_np.min()
        label_pat_np = ndimage.zoom(label_pat_np, factor, order=self.scale_order) # bilinear resample
        # eliminate illegal values
        if self.scale_order >= 2:
            bigger = label_pat_np>pmax
            smaller = label_pat_np<pmin
            if bigger.any() or smaller.any():
                label_pat_np[bigger] = pmax
                label_pat_np[smaller] = pmin
        
        img_patch = torch.from_numpy(img_pat_np).float()
        label_patch = torch.from_numpy(label_pat_np).long()
        img_patch, label_patch = crop(img_patch, (0,0,*crop_size)), crop(label_patch, (0,0,*crop_size))
        # crop again to eliminate extra pixels
        return img_patch, label_patch


class RandomRotate2d(object):
    def __init__(self):
        pass

    def __call__(self, img, label):
        k = random.randint(0, 3)
        if k == 0:
            return img, label
        return rot90_tensor(img, k), rot90_tensor(label, k)


class RandomDropout(object):
    """ Randomly drop an input channel / modality """
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate

    def __call__(self, tensor, label):
        if self.drop_rate <= 0:
            return tensor, label
        elif self.drop_rate > 1:
            raise RuntimeError('Dropout rate greater than 1.')

        C = tensor.size(0)
        drop_count = 0
        rand_flag = np.random.random(C)
        rand_flag = rand_flag < self.drop_rate
        if all(rand_flag):
            rand_flag[random.randint(0, C-1)] = False
        for c in range(C):
            if rand_flag[c]:
                drop_count += 1
                tensor[c, ...] = 0.0
#        print(rand_flag)
#        print(drop_count)
        tensor *= C / (C-drop_count)
        return tensor, label


class RandomBlack(object):
    """ Randomly set patches to zeor"""
    def __init__(self, black_patch_size=None):
        self.black_patch_size = black_patch_size
        if black_patch_size is None:
            self.blacker = passthrough
        elif len(black_patch_size) == 2:
            self.blacker = RandomBlack2d(black_patch_size)
        elif len(black_patch_size) == 3:
            self.blacker = RandomBlack3d(black_patch_size)
        else:
            raise RuntimeError('Invalid length of black_patch_size.')
        
    def __call__(self, tensor, label):
        return self.blacker(tensor, label)
        

class RandomBlack2d(object):
    """ Randomly set patches to zeor"""
    def __init__(self, black_patch_size=None):
        self.black_patch_size = black_patch_size
        
    def __call__(self, tensor, label):
        th, tw = self.black_patch_size
        h, w = tensor.size()[-2:]
        x1 = random.randint(0, h-th)
        y1 = random.randint(0, w-tw)
        tensor[..., x1:x1+th, y1:y1+tw] = 0
        label[..., x1:x1+th, y1:y1+tw] = 0
        
        return tensor, label


class RandomBlack3d(object):
    """ Randomly set patches to zero"""
    def __init__(self, black_patch_size=None):
        self.black_patch_size = black_patch_size
        
    def __call__(self, tensor, label):
        td, th, tw = self.black_patch_size
        d, h, w = tensor.size()[-3:]
        x1 = random.randint(0, d-td)
        y1 = random.randint(0, h-th)
        z1 = random.randint(0, w-tw)
        tensor[..., x1:x1+td, y1:y1+th, z1:z1+tw] = 0
#        label[..., x1:x1+td, y1:y1+th, z1:z1+tw] = 0
        
        return tensor, label

class RandomNoise(object):
    """ Randomly add Gaussian noise"""
    def __init__(self, prob, max_scale=0.3):
        self.prob = prob
        self.max_scale = max_scale
    
    def __call__(self, tensor, label):
        if random.random() < self.prob:
            scale = self.max_scale * random.random()
            noise = torch.randn(*tensor.shape) * scale
            tensor += noise
        return tensor, label

    
def image_to_patch(image, patch_sz, overlap):
    """ Split a 2D/3D image into overlapped patches"""
    if len(image.size()) == 2:
        pass
    
def image_to_patch3d(images, patch_sz, overlap):
    """ Split a batch of channelled 3D images into overlapped patches"""
    if patch_sz is None or overlap is None:
        return images
    if isinstance(patch_sz, int):
        patch_sz = (patch_sz,)*3
    if isinstance(overlap, int):
        overlap = (overlap,)*3
    # image should be a tensor of size N x C x D x H x W
    d, h, w = images.size()[-3:]
    l = list(range(max(d, h, w)))
    start_pos_d = l[0:d-patch_sz[0]:patch_sz[0]-overlap[0]] + [d-patch_sz[0]]
    start_pos_h = l[0:h-patch_sz[1]:patch_sz[1]-overlap[1]] + [h-patch_sz[1]]
    start_pos_w = l[0:w-patch_sz[2]:patch_sz[2]-overlap[2]] + [w-patch_sz[2]]
    patch_list = []
    psz_np = np.array(patch_sz)
    for i in start_pos_d:
        for j in start_pos_h:
            for k in start_pos_w:
                sp = np.array([i, j, k])
                ep = sp + psz_np
                if any(ep > np.array([d, h, w])):
                    raise RuntimeError('ep > size')
                patch_list.append(images[..., sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]])
    
    return patch_list
        
def patch_to_image3d(images, patch_list, patch_sz, overlap):
    """ Stitch overlapped patches into a batch of channelled 3D images"""
    device = patch_list[0].device
    if patch_sz is None or overlap is None:
        return images
    if isinstance(patch_sz, int):
        patch_sz = (patch_sz,)*3
    if isinstance(overlap, int):
        overlap = (overlap,)*3
    # images should be a tensor of size N x C x D x H x W
    d, h, w = images.size()[-3:]
    l = list(range(max(d, h, w)))
    start_pos_d = l[0:d-patch_sz[0]:patch_sz[0]-overlap[0]] + [d-patch_sz[0]]
    start_pos_h = l[0:h-patch_sz[1]:patch_sz[1]-overlap[1]] + [h-patch_sz[1]]
    start_pos_w = l[0:w-patch_sz[2]:patch_sz[2]-overlap[2]] + [w-patch_sz[2]]
    shape = patch_list[0].size()[:-3] + images.size()[-3:]
    try:
        recon_image = torch.zeros(shape, dtype=patch_list[0].dtype, device=device)
        overlap_counter = torch.zeros_like(recon_image, dtype=torch.uint8)
        patch_pad = torch.zeros_like(recon_image)
    except:
        print('Stitching on CPU.')
        recon_image = torch.zeros(shape, dtype=patch_list[0].dtype, device='cpu')
        overlap_counter = torch.zeros_like(recon_image, dtype=torch.uint8)
        patch_pad = torch.zeros_like(recon_image)
    psz_np = np.array(patch_sz)
    patch_index = 0
    for i in start_pos_d:
        for j in start_pos_h:
            for k in start_pos_w:
                sp = np.array([i, j, k])
                ep = sp + psz_np
                patch_pad[..., sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] = patch_list[patch_index]
                recon_image += patch_pad
                if overlap_counter.max() == 255 and overlap_counter.dtype is torch.uint8:
                    overlap_counter = overlap_counter.type(torch.int16)
                overlap_counter[..., sp[0]:ep[0], sp[1]:ep[1], sp[2]:ep[2]] += 1
                patch_index += 1 
                patch_pad.fill_(0)
    recon_image /= overlap_counter
    
    return recon_image.to(device)
        
def image_to_patch2d(images, patch_sz, overlap):
    """ Split a batch of channelled 2D images into overlapped patches"""
    if patch_sz is None or overlap is None:
        return images
    if isinstance(patch_sz, int):
        patch_sz = (patch_sz,)*2
    if isinstance(overlap, int):
        overlap = (overlap,)*2
    # image should be a tensor of size N x C x H x W
    h, w = images.size()[-2:]
    l = list(range(max(h, w)))
    start_pos_h = l[0:h-patch_sz[0]:patch_sz[0]-overlap[0]] + [h-patch_sz[0]]
    start_pos_w = l[0:w-patch_sz[1]:patch_sz[1]-overlap[1]] + [w-patch_sz[1]]
    patch_list = []
    psz_np = np.array(patch_sz)
    for j in start_pos_h:
        for k in start_pos_w:
            sp = np.array([j, k])
            ep = sp + psz_np
            if any(ep > np.array([h, w])):
                raise RuntimeError('ep > size')
            patch_list.append(images[..., sp[0]:ep[0], sp[1]:ep[1]])
    
    return patch_list
        
def patch_to_image2d(images, patch_list, patch_sz, overlap):
    """ Stitch overlapped patches into a batch of channelled 2D images"""
    if patch_sz is None or overlap is None:
        return images
    if isinstance(patch_sz, int):
        patch_sz = (patch_sz,)*2
    if isinstance(overlap, int):
        overlap = (overlap,)*2
    # images should be a tensor of size N x C x H x W
    h, w = images.size()[-2:]
    l = list(range(max(h, w)))
    start_pos_h = l[0:h-patch_sz[0]:patch_sz[0]-overlap[0]] + [h-patch_sz[0]]
    start_pos_w = l[0:w-patch_sz[1]:patch_sz[1]-overlap[1]] + [w-patch_sz[1]]
    shape = patch_list[0].size()[:-2] + images.size()[-2:]
    recon_image = torch.zeros(shape, dtype=patch_list[0].dtype, device=patch_list[0].device)
    overlap_counter = torch.zeros_like(recon_image, dtype=torch.float32)
    psz_np = np.array(patch_sz)
    patch_index = 0
    for j in start_pos_h:
        for k in start_pos_w:
            sp = np.array([j, k])
            ep = sp + psz_np
            patch_pad = torch.zeros_like(recon_image)
            patch_pad[..., sp[0]:ep[0], sp[1]:ep[1]] = patch_list[patch_index]
            recon_image += patch_pad
            overlap_counter[..., sp[0]:ep[0], sp[1]:ep[1]] += 1
            patch_index += 1 
    recon_image /= overlap_counter
    
    return recon_image    
    
    
