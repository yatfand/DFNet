#!/usr/bin/python3
#coding=utf-8
import cv2
import torch
import numpy as np
import random
import math
import collections


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops
    def __call__(self, image, depth, mask, boundary):
        for op in self.ops:
            image, depth, mask, boundary = op(image, depth, mask, boundary)
        return image, depth, mask, boundary

class Normalize(object):
    def __init__(self, mean, std, d_mean=None, d_std=None):
        self.mean = mean
        self.std  = std
        self.d_mean = d_mean
        self.d_std = d_std
        if self.d_mean is None:
            print("Warning: no d_mean supplied in Normalize!")
            self.d_mean = np.mean(self.mean)
            self.d_std = np.mean(self.std)

    def __call__(self, image, depth, mask, boundary):
        mask /= 255.0
        image /= 255.0
        depth /= 255.0
        boundary /= 255.0
        return image, depth,  mask, boundary

class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = size
            self.W = self.size
            self.H = self.size
            self.keep_ratio = True
        else:
            H, W = size
            self.H = H
            self.W = W
            self.keep_ratio = False

    def __call__(self, image, depth, mask, boundary):
        if not self.keep_ratio:
            image = self.resize(image, self.H, self.W)
            depth = self.resize(depth, self.H, self.W)
            mask  = self.resize(mask,  self.H, self.W)
            boundary = self.resize(boundary, self.H, self.W)
        else:
            h, w, _ = image.shape
            if w < h:
                ow = self.size
                oh = int(self.size*h/w)
                image = self.resize(image, oh, ow)
                depth = self.resize(depth, oh, ow)
                mask  = self.resize(mask,  oh, ow)
                boundary = self.resize(boundary, oh, ow)
            else:
                oh = self.size
                ow = int(self.size*w/h)
                image = self.resize(image, oh, ow)
                depth = self.resize(depth, oh, ow)
                mask  = self.resize(mask , oh, ow)
                boundary = self.resize(boundary, oh, ow)
            h, w, _ = image.shape
            if h > 400:
                print("image.shape:{}".format(image.shape))
                image = self.resize(image, 400, self.size)
                depth = self.resize(depth, 400, self.size)
                mask  = self.resize(mask , 400, self.size)
                boundary = self.resize(boundary, 400, self.size)
            elif w > 400:
                print("image.shape:{}".format(image.shape))
                image = self.resize(image, self.size, 400)
                depth = self.resize(depth, self.size, 400)
                mask  = self.resize(mask , self.size, 400)
                boundary = self.resize(boundary, self.size, 400)
        return image, depth, mask, boundary
    def resize(self, img, h, w):
        return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

class MultiResize(object):
    def __init__(self, size_list):
        assert isinstance(size_list, list)
        self.size_list = size_list
    def __call__(self, image, depth, mask, boundary):
        images, depths, masks, boundarys = [], [], [], []
        for size in self.size_list:
            image = self.resize(image, size, size)
            depth = self.resize(depth, size, size)
            mask  = self.resize(mask , size, size)
            boundary = self.resize(boundary, size, size)
            images.append(image)
            depths.append(depth)
            masks.append(mask)
            boundarys.append(boundary)
        return images, depths, masks, boundarys

    def resize(self, img, h, w):
        return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

class  MultiNormalize(object):
    def __call__(self, images, depths, masks, boundarys):
        for i in range(len(images)):
            images[i] /= 255.0
            depths[i] /= 255.0
            masks[i] /= 255.0
            boundarys[i] /= 255.0
        return images, depths, masks, boundarys

class MultiToTensor(object):
    def __call__(self, images, depths, masks, boundarys):
        images_t, depths_t, masks_t, boundarys_t = [], [], [], []
        for i in range(len(images)):
            image = torch.from_numpy(images[i])
            image = image.permute(2, 0, 1)
            mask  = torch.from_numpy(masks[i])
            mask  = mask.permute(2, 0, 1).mean(dim=0, keepdim=True)
            depth = torch.from_numpy(depths[i])
            depth = depth.permute(2, 0, 1).mean(dim=0, keepdim=True)
            boundary = torch.from_numpy(boundarys[i])
            boundary = boundary.permute(2, 0, 1).mean(dim=0, keepdim=True)
            images_t.append(image)
            depths_t.append(depth)
            masks_t.append(mask)
            boundarys_t.append(boundary)

        return images_t, depths_t, masks_t, boundarys_t

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W
    def __call__(self, image, depth, mask, boundary):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        boundary = boundary[ymin:ymin + self.H, xmin:xmin + self.W, :]
        depth = depth[ymin:ymin+self.H, xmin:xmin+self.W]
        return image, depth,  mask, boundary

def random_hflip(image, depth, mask, boundary):
    if np.random.randint(2)==1:
        image = image[:,::-1,:].copy()
        mask  =  mask[:,::-1,:].copy()
        depth = depth[:, ::-1, :].copy()
        boundary = boundary[:, ::-1, :].copy()
    return image, depth,  mask, boundary

class RandomHorizontalFlip(object):
    def __call__(self, image,depth,  mask, boundary):
        return random_hflip(image, depth, mask, boundary)

class MultiRandomHorizontalFlip(object):
    def __call__(self, images, depths, masks, boundarys):
        len_ = len(images)
        for i in range(len_):
            images[i], depths[i], masks[i], boundarys[i] = random_hflip(images[i], depths[i], masks[i], boundarys[i])
        return images, depths, masks, boundarys

class ToTensor(object):
    def __init__(self, depth_gray=True):
        self.depth_gray = depth_gray
    def __call__(self, image, depth, mask, boundary):
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = np.ascontiguousarray(mask)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        depth = np.ascontiguousarray(depth)
        depth = torch.from_numpy(depth)
        depth = depth.permute(2, 0, 1)
        boundary = np.ascontiguousarray(boundary)
        boundary = torch.from_numpy(boundary)
        boundary = boundary.permute(2, 0, 1)
        if self.depth_gray == True:
            return image, depth.mean(dim=0, keepdim=True), mask.mean(dim=0, keepdim=True), boundary.mean(dim=0, keepdim=True)
        else:
            return image, depth, mask.mean(dim=0, keepdim=True), boundary.mean(dim=0, keepdim=True)

class RandomMask(object):
    def __init__(self):
        self.thresh = 1
    def out_saliency(self, x1, y1, x2, y2, gt):
        return gt[y1:y2, x1:x2].sum() == 0
    def __call__(self, image, depth, gt):
        if True:
            s = image.shape
            ht = s[0]
            wd = s[1]
            grid_sizes=[0,16,32,44,56]
            hide_prob = 0.5
            grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]
            if(grid_size > 0):
                for x in range(0,wd,grid_size):
                    for y in range(0,ht,grid_size):
                        x_end = min(wd, x+grid_size)
                        y_end = min(ht, y+grid_size)
                        if (random.random() <=  hide_prob) and self.out_saliency(x, y, x_end, y_end, gt):
                            image[y:y_end,x:x_end,:]=0
                            depth[y:y_end, x:x_end, :] = 0
            return image, depth, gt
        else:
            return image, depth, gt

def resize(img, size, interpolation=cv2.INTER_CUBIC):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    w, h, =  size
    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    else:
        output = cv2.resize(img, dsize=size[::-1], interpolation=interpolation)
    return output
def crop(img, i, j, h, w):
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if len(img.shape) == 3:
        return img[i:i+h, j:j+w, :]
    else:
        return img[i:i+h, j:j+w]

def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_CUBIC):
    assert _is_numpy_image(img), 'img should be numpy image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation=interpolation)
    return img

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_CUBIC):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img, depth, mask, boundary):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img_1 = resized_crop(img, i, j, h, w, self.size, self.interpolation)
        depth_1 = resized_crop(depth, i, j, h, w, self.size, self.interpolation)
        mask_1 = resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        boundary_1 = resized_crop(boundary, i, j, h, w, self.size, self.interpolation)
        return img_1, depth_1, mask_1, boundary_1

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string