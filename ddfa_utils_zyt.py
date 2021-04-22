#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data
import cv2, skimage
import pickle
import argparse
import matplotlib.pyplot as plt
from io_utils import _numpy_to_tensor, _load_cpu, _load_gpu
import scipy.io as scio
import matplotlib.image as mping
import random
# from PIL import Image
import pdb
def img_loader(path, img_size=64):
    #print(path)
    #img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    img = cv2.imread(path)
    #print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, (112, 96), interpolation=cv2.INTER_CUBIC)
    return img
    # return


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DDFADataset(data.Dataset):
    def __init__(self, root, img_size, filelists, transform=None, **kargs):# lms_fp, file_dic_fp, ids_dic, ori_fp, transform=None, **kargs): # zyt add
        self.root = root
        #pdb.set_trace()
        self.img_size = img_size
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.file_dict, self.id_mapping_dict = self._load_id_dict()
        # import pdb; pdb.set_trace()
        # self.lms = _load_cpu(lms_fp)
        # self.dic = _load_cpu(file_dic_fp)
        # self.dic_ids = _load_cpu(ids_dic)
        # self.oriRoot = ori_fp
        
        self.img_loader = img_loader
    
    def _load_id_dict(self):
        file_dict = {}
        id_mapping_dict = {}
        for _line in self.lines:
            key, value = int(_line.split()[1]), _line.split()[0]
            id_mapping_dict[value.split('/')[0]] = key
            if key in file_dict.keys():
                file_dict[key].append(value)
            else:
                file_dict[key] = []
        return file_dict, id_mapping_dict


    # def _lms_loader(self, index):
    #     # print(index)
    #     lms = self.lms[index]

    #     return lms

    # def _dic_loader(self, path):
    #     aa = path.strip().split('/')[-1]
    #     sor_img = self.dic[aa]

    #     return sor_img

    # def _dic_loader_ids(self, path):
    #     sor_id = self.dic_ids[path]

    #     return sor_id

    def __getitem__(self, index):
        src_name = self.lines[index].split()[0]
        src_path = osp.join(self.root, src_name)
        src_img = self.img_loader(src_path, self.img_size)
        src_id = self.id_mapping_dict[src_name.split('/')[0]]
        # lms = self._lms_loader(index)

        dst_name = random.choice(self.file_dict[src_id])
        dst_path = osp.join(self.root, dst_name)
        dst_img = self.img_loader(dst_path, self.img_size)

        if self.transform is not None:
            src_img = self.transform(src_img)
            dst_img = self.transform(dst_img)
        return src_img, src_name, dst_img, dst_name

    def __len__(self):
        return len(self.lines)

if __name__ == "__main__":
    dataset = DDFADataset("./aqy_train_120x120", 64, "./aqy_train_120x120/train_aqy_list_sub.txt")
    dataset.__getitem__(0)
# class DDFATestDataset(data.Dataset):
#     def __init__(self, filelists, img_size, root='', transform=None):
#         self.root = root
#         self.img_size = img_size
#         self.transform = transform
#         self.lines = Path(filelists).read_text().strip().split('\n')

#     def __getitem__(self, index):
#         path = osp.join(self.root, self.lines[index])
#         img = img_loader(path, self.img_size)

#         if self.transform is not None:
#             img = self.transform(img)
#         return img

#     def __len__(self):
#         return len(self.lines)
