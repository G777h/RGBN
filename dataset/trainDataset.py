import glob
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, train_dir, is_train, channel=4, debug=False):
        self.image_size = 256
        self.train = is_train
        self.channel = channel
        if channel > 1:
            rgb_dir = train_dir + "/color/*.tif*"
            print("rgb_dir", rgb_dir)
            self.rgb_files = sorted(glob.glob(rgb_dir))
            if debug:
                self.rgb_files = self.rgb_files[:100]

            self.len = len(self.rgb_files)

        if channel == 1 or channel == 4 or channel == 6:
            depth_dir = train_dir + "/gt/*.tif*"
            print("depth_dir", depth_dir)
            self.depth_files = sorted(glob.glob(depth_dir))
            # self.depth_max = 255
            if debug:
                self.depth_files = self.depth_files[:100]
            self.len = len(self.depth_files)
            
    def __getitemForChannel6__(self, index):
        rgb_path = self.rgb_files[index]
        normal_path = self.depth_files[index]

        # 1. 使用 cv2 读取 RGB (保留原始位深)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if rgb_img is not None and rgb_img.ndim == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # tif 默认 BGR 读取，转回 RGB
        
        # 2. 使用 cv2 读取 Normal 法向图
        normal_img = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        if normal_img is not None and normal_img.ndim == 3:
            normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)

        # 3. 动态归一化 (兼容 8-bit 的 0-255 和 16-bit 的 0-65535)
        rgb_img = rgb_img.astype(np.float32)
        rgb_img /= 65535.0 if rgb_img.max() > 255.0 else 255.0

        normal_img = normal_img.astype(np.float32)
        normal_img /= 65535.0 if normal_img.max() > 255.0 else 255.0

        # 4. 转换维度 (H,W,C) -> (C,H,W) 并转为 Tensor
        rgb_img = rgb_img.transpose(2, 0, 1)
        normal_img = normal_img.transpose(2, 0, 1)

        rgb = torch.from_numpy(rgb_img).type(torch.FloatTensor)
        normal = torch.from_numpy(normal_img).type(torch.FloatTensor)

        # 5. 空间对齐的数据增强
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(256, 256))
            rgb = TF.crop(rgb, i, j, h, w)
            normal = TF.crop(normal, i, j, h, w)
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                normal = TF.hflip(normal)
            if random.random() > 0.5:
                rgb = TF.vflip(rgb)
                normal = TF.vflip(normal)
        else:
            transform = transforms.CenterCrop((448, 576))
            rgb = transform(rgb)
            normal = transform(normal)
            
        return rgb, normal        

    def __getitemForChannel4__(self, index):
        rgb_path = self.rgb_files[index]
        img = Image.open(rgb_path).convert("RGB")
        img = np.array(img) / 255
        img = img.transpose(2, 0, 1)

        depth_path = self.depth_files[index]
        depth = Image.open(depth_path)
        depth_max = 255.0 if np.array(depth).max() < 255 else self.depth_max
        depth = np.array(depth) / depth_max
        if len(depth.shape) == 3:
            depth = depth[0]

        rgb = torch.from_numpy(img)  # [3,H,W]
        depth = torch.from_numpy(depth)  # [H,W]
        depth = torch.unsqueeze(depth, 0)

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(256, 256))
            rgb = TF.crop(rgb, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
            # Random vertical flipping
            if random.random() > 0.5:
                rgb = TF.vflip(rgb)
                depth = TF.vflip(depth)
        else:
            transform = transforms.CenterCrop((448, 576))
            rgb = transform(rgb)
            depth = transform(depth)
        rgb = rgb.type(torch.FloatTensor)
        depth = depth.type(torch.FloatTensor)
        return rgb, depth

    def __getitemForChannel3__(self, index):
        rgb_path = self.rgb_files[index]
        img = Image.open(rgb_path).convert("RGB")
        img = np.array(img) / 255
        img = img.transpose(2, 0, 1)
        rgb = torch.from_numpy(img)  # [3,H,W]

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(256, 256))
            rgb = TF.crop(rgb, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
            # Random vertical flipping
            if random.random() > 0.5:
                rgb = TF.vflip(rgb)
        else:
            transform = transforms.CenterCrop((448, 576))
            rgb = transform(rgb)
        rgb = rgb.type(torch.FloatTensor)
        return rgb

    def __getitemForChannel1__(self, index):
        depth_path = self.depth_files[index]
        depth = Image.open(depth_path)
        depth_max = 255 if np.array(depth).max() < 255 else self.depth_max
        depth = np.array(depth) / depth_max
        if len(depth.shape) == 3:
            depth = depth[0]

        depth = torch.from_numpy(depth)  # [H,W]
        depth = torch.unsqueeze(depth, 0)

        # Random crop
        # top: int, left: int, height: int, width: int
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(depth, output_size=(256, 256))
            depth = TF.crop(depth, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                depth = TF.hflip(depth)
            # Random vertical flipping
            if random.random() > 0.5:
                depth = TF.vflip(depth)
        else:
            transform = transforms.CenterCrop((448, 576))
            depth = transform(depth)
        depth = depth.type(torch.FloatTensor)
        return depth

    def __getitem__(self, index):
        if self.channel == 6:
            return self.__getitemForChannel6__(index)
        if self.channel == 4:
            return self.__getitemForChannel4__(index)
        if self.channel == 3:
            return self.__getitemForChannel3__(index)
        if self.channel == 1:
            return self.__getitemForChannel1__(index)

    def __len__(self):
        return self.len


class nyuv2(BaseDataset):
    def __init__(self, train_dir, is_train, channel=4, debug=False):
        super().__init__(train_dir, is_train, channel, debug)
        self.depth_max = 10000.0


class sun(BaseDataset):
    def __init__(self, train_dir, is_train, channel=4, debug=False):
        super().__init__(train_dir, is_train, channel, debug)
        self.depth_max = 100000.0
