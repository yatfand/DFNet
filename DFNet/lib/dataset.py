#!/usr/bin/python3
#coding=utf-8

import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset, DataLoader


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        print('Parameters...\n')
        for k, v in self.kwargs.items():
            print('️~ %-10s: %s'%(k, v))
        self.mean = np.array([[[128.67, 117.24, 107.97]]])
        self.std = np.array([[[66.14, 58.32, 56.37]]])
        self.d_mean = 116.09
        self.d_std = 56.61

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

class RGBDData(Dataset):
    def __init__(self, cfg):
        self.samples = []
        self.mode = cfg.mode
        if cfg.mode == "train":
            with open(osp.join(cfg.datapath, "TR.txt"), "r") as fin:
                for line in fin:
                    line = line.strip()
                    image_name = osp.join(cfg.datapath, "TR/RGB", line + ".png")
                    depth_name = osp.join(cfg.datapath, "TR/depth", line + ".png")
                    mask_name = osp.join(cfg.datapath, "TR/GT", line + ".png")
                    boundary_name = osp.join(cfg.datapath, "TR/Boundary", line + "_edge.png")
                    self.samples.append([line, image_name, depth_name, mask_name, boundary_name])
            print("train mode: len(samples):%s"%(len(self.samples)))
        
        elif cfg.mode == "test":
            with open(osp.join(cfg.datapath, "TE.txt"), "r") as lines:
                for line in lines:
                    line = line.strip()
                    image_name = osp.join(cfg.datapath, "TE/RGB", line + ".png")
                    depth_name = osp.join(cfg.datapath, "TE/depth", line + ".png")
                    mask_name = osp.join(cfg.datapath, "TE/GT", line + ".png")
                    boundary_name = osp.join(cfg.datapath, "TE/GT", line + ".png")
                    self.samples.append([line, image_name, depth_name, mask_name, boundary_name])
            print("test mode: len(samples):%s" % (len(self.samples)))
            
        elif cfg.mode == "val":
            with open(osp.join(cfg.datapath, "VAL.txt"), "r") as lines:
                for line in lines:
                    line = line.strip()
                    image_name = osp.join(cfg.datapath, "VAL/RGB", line + ".png")
                    depth_name = osp.join(cfg.datapath, "VAL/depth", line + ".png")
                    mask_name = osp.join(cfg.datapath, "VAL/GT", line + ".png")
                    boundary_name = osp.join(cfg.datapath, "VAL/Boundary", line + "_edge.png")
                    self.samples.append([line, image_name, depth_name, mask_name, boundary_name])
            print("val mode: len(samples):%s" % (len(self.samples)))
        
        else:
            print('error！')

        if cfg.mode == 'train':
            if cfg.train_scales is None:
                cfg.train_scales = [224, 256, 320]
            print("Train_scales:", cfg.train_scales)
            self.transform = transform.Compose(transform.MultiResize(cfg.train_scales),
                                                transform.MultiRandomHorizontalFlip(),
                                                transform.MultiNormalize(),
                                                transform.MultiToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Resize((256, 256)),
                                                transform.Normalize(mean=cfg.mean, std=cfg.std, d_mean=cfg.d_mean, d_std=cfg.d_std),
                                                transform.ToTensor(depth_gray=True))
        elif cfg.mode == 'val':
            self.transform = transform.Compose(transform.Resize((256, 256)),
                                                transform.Normalize(mean=cfg.mean, std=cfg.std, d_mean=cfg.d_mean, d_std=cfg.d_std),
                                                transform.ToTensor(depth_gray=True))
        else:
            raise ValueError

    def __getitem__(self, idx):
        key, image_name, depth_name, mask_name, boundary_name = self.samples[idx]
        image               = cv2.imread(image_name).astype(np.float32)[:,:,::-1]
        depth               = cv2.imread(depth_name).astype(np.float32)[:,:, ::-1]
        mask                = cv2.imread(mask_name).astype(np.float32)[:,:,::-1]
        boundary            = cv2.imread(boundary_name).astype(np.float32)[:, :, ::-1]
        H, W, C             = mask.shape
        image, depth, mask, boundary = self.transform(image, depth, mask, boundary)

        if self.mode == "train":
            gate_gt = torch.zeros(1)
            key1 = int(key)
            gate_gt[0] = key1
            return image, depth, mask, boundary, gate_gt
        elif self.mode == 'test':
            mask_name = mask_name.split("/")[-1]
            return image, depth, mask, (H,W), mask_name, boundary
        elif self.mode == 'val':
            mask_name = mask_name.split("/")[-1]
            return image, depth, mask, (H,W), mask_name, boundary
        else:
            print('getitem error')
    def __len__(self):
        return len(self.samples)

def train_collate_fn(batch):
    images, depths, masks, boundarys, gate_gt = zip(*batch)
    l = len(images[0])
    images_t, depths_t, masks_t, boundarys_t = {}, {}, {}, {}
    gates_t = {}
    gate_gt = torch.stack(gate_gt)
    for i in range(l):
        images_t[i] = []
        depths_t[i] = []
        masks_t[i] = []
        boundarys_t[i] = []
        gates_t[i] = gate_gt

    for i in range(len(images)):
        for j in range(l):
            images_t[j].append(images[i][j])
            depths_t[j].append(depths[i][j])
            masks_t[j].append(masks[i][j])
            boundarys_t[j].append(boundarys[i][j])

    for i in range(l):
        images_t[i] = torch.stack(images_t[i])
        depths_t[i] = torch.stack(depths_t[i])
        masks_t[i] = torch.stack(masks_t[i])
        boundarys_t[i] = torch.stack(boundarys_t[i])
    return images_t, depths_t, masks_t, boundarys_t, gates_t