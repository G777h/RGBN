import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    # 新增 split_name 参数，显式指定要读取的子文件夹名称
    def __init__(self, root="/data1/Qihao_data/data/pswild", transform=None, channel=3, debug=False, split_name="color"):
        if channel == 3:
            self.mode = "RGB"
        elif channel == 1:
            self.mode = "L"

        self.split = split_name
        splitdir = Path(root) / self.split
        print(f"imagefolder: splitdir {splitdir}")
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{splitdir}"')

        # 修改后缀过滤，同时兼容 png, tif, tiff
        self.samples = [
            f for f in splitdir.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.png', '.tif', '.tiff']
        ]
        self.samples.sort()  
        if debug:
            self.samples = self.samples[:20]

        self.transform = transform

    def __getitem__(self, index):
        imgname = str(self.samples[index])
        # cv2.IMREAD_UNCHANGED 同样完美支持 8-bit 和 16-bit 的 PNG
        img = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Failed to load image: {imgname}")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 动态归一化 (兼容 8-bit 和 16-bit)
        img = img.astype(np.float32)
        img /= 65535.0 if img.max() > 255.0 else 255.0

        if self.mode == "RGB":
            # 通道转换 (H, W, C) -> (C, H, W)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
        elif self.mode == "L":
            img = np.expand_dims(img, 0).astype("float32")
            img = torch.from_numpy(img).float()

        file_name = os.path.basename(imgname)
        file_name_without_extension = os.path.splitext(file_name)[0]
        return img, file_name_without_extension

    def __len__(self):
        return len(self.samples)


class ImageFolderUnited(Dataset):
    def __init__(self, root="/data/xyy/nyu5k/nyuv2/test", transform=None, debug=False):
        # 显式传入 split_name="color" 和 split_name="gt"，不再依赖 channel 参数判断文件夹
        self.rgb_dataloader = ImageFolder(root=root, transform=transform, channel=3, debug=debug, split_name="color")
        self.depth_dataloader = ImageFolder(root=root, transform=transform, channel=3, debug=debug, split_name="gt")

    def __getitem__(self, index):
        rgb, rgb_path = self.rgb_dataloader.__getitem__(index)
        depth, depth_path = self.depth_dataloader.__getitem__(index)
        return rgb, depth, rgb_path, depth_path

    def __len__(self):
        return len(self.rgb_dataloader)