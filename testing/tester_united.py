import os
import time
import numpy as np
import cv2
import torch
from dataset.testDataset import ImageFolderUnited
from dataset.utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.IOutils import *
from utils.metrics import AverageMeter, compute_metrics

from .tester_single import TesterSingle


class TesterUnited(TesterSingle):
    def __init__(self, args, model_config) -> None:
        super().__init__(args, model_config)

    def init_dataset(self, test_dataset, test_batch_size, num_workers, channel):
        test_transforms = transforms.Compose([transforms.ToTensor()])
        test_dataset = ImageFolderUnited(test_dataset, transform=test_transforms, debug=self.debug)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False)
        return test_dataloader

    def getAvgMeter(self):
        return {
            "avg_rgb_psnr": AverageMeter(),
            "avg_rgb_ms_ssim": AverageMeter(),
            "avg_rgb_bpp": AverageMeter(),
            "avg_depth_psnr": AverageMeter(),
            "avg_depth_ms_ssim": AverageMeter(),
            "avg_depth_bpp": AverageMeter(),
            "avg_deocde_time": AverageMeter(),
            "avg_encode_time": AverageMeter(),
        }

    def updateAvgMeter(self, avgMeter, rgb_p, rgb_m, rgb_bpp, depth_p, depth_m, depth_bpp, dec_time, enc_time):
        avgMeter["avg_rgb_psnr"].update(rgb_p)
        avgMeter["avg_rgb_ms_ssim"].update(rgb_m)
        avgMeter["avg_rgb_bpp"].update(rgb_bpp)
        avgMeter["avg_depth_psnr"].update(depth_p)
        avgMeter["avg_depth_ms_ssim"].update(depth_m)
        avgMeter["avg_depth_bpp"].update(depth_bpp)
        avgMeter["avg_deocde_time"].update(dec_time)
        avgMeter["avg_encode_time"].update(enc_time)

    @torch.no_grad()
    def test_model(self, padding_mode="reflect0", padding=True):
        self.net.eval()
        avgMeter = self.getAvgMeter()
        rec_dir = self.get_rec_dir(padding=padding, padding_mode=padding_mode)

        for i, (rgb, depth, rgb_img_name, depth_img_name) in enumerate(self.test_dataloader):
            B, C, H, W = rgb.shape

            rgb = rgb.to(self.device)
            depth = depth.to(self.device)

            rgb_pad = pad(rgb, padding_mode)
            depth_pad = pad(depth, padding_mode)
            rgb_stream_path = os.path.join(rec_dir, "depth_bin")
            depth_stream_path = os.path.join(rec_dir, "rgb_bin")
            rgb_bpp, depth_bpp, enc_time = self.compress_one_image_united(
                x=(rgb_pad, depth_pad),
                stream_path=(rgb_stream_path, depth_stream_path),
                H=H,
                W=W,
                img_name=rgb_img_name[0],
            )
            rgb_x_hat, depth_x_hat, dec_time = self.decompress_one_image_united(
                stream_path=(rgb_stream_path, depth_stream_path), img_name=rgb_img_name[0], mode=padding_mode
            )
            self.test_save_and_log_perimg(
                i,
                rgb_x_hat,
                depth_x_hat,
                rgb,
                depth,
                rec_dir,
                rgb_img_name,
                avgMeter,
                rgb_bpp,
                depth_bpp,
                dec_time,
                enc_time,
            )
        self.test_finish_log(avgMeter, rec_dir)

    

    def test_save_and_log_perimg(
        self, i, rgb_x_hat, depth_x_hat, rgb, depth, rec_dir, img_name, avgMeter, rgb_bpp, depth_bpp, dec_time, enc_time
    ):
        rgb_p, rgb_m = compute_metrics(rgb_x_hat, rgb)
        depth_p, depth_m = compute_metrics(depth_x_hat, depth)
        r_bpp_psnr = f"{rgb_bpp:.4f}_{rgb_p:.4f}_"
        d_bpp_psnr = f"{depth_bpp:.4f}_{depth_p:.4f}_"

        # 1. 保存 RGB (通常是 8-bit, saveImg 内部应该处理了)
        saveImg(rgb_x_hat, os.path.join(rec_dir, "rgb_rec", f"{img_name[0]}_{r_bpp_psnr}_rec.png"))
        
        # 2. 保存 Normal Map (替代原本的 depth_x_hat)
        # 同样使用原有的 saveImg 处理 8-bit 保存
        saveImg(depth_x_hat, os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_{d_bpp_psnr}_rec_8bit.png"))

        # 3. 处理 16-bit 的 Normal Map 保存 (这里是原先报错的地方)
        if rec_dir.find("sun") != -1:
            depth_16bit = depth_x_hat * 100000
        else:
            # 一般归一化恢复
            depth_16bit = depth_x_hat * 65535.0  # 之前我们改为了除以 65535，这里需要对应乘回来
            
        # [核心修复]: 处理 3 通道 Tensor
        # 1. 取出第一张图 [0]
        # 2. 转换设备并转换为 NumPy 数组
        # 3. 限制数值范围并转为 uint16
        # 4. Transpose (C, H, W) -> (H, W, C)
        depth_numpy = depth_16bit[0].cpu().numpy()
        depth_numpy = np.clip(depth_numpy, 0, 65535).astype(np.uint16)
        
        if depth_numpy.ndim == 3 and depth_numpy.shape[0] == 3:
            depth_numpy = np.transpose(depth_numpy, (1, 2, 0))
            # 转为 BGR 供 OpenCV 正确保存颜色
            depth_numpy = cv2.cvtColor(depth_numpy, cv2.COLOR_RGB2BGR)
        elif depth_numpy.ndim == 3 and depth_numpy.shape[0] == 1:
            # 兼容如果确实输入了单通道深度图的情况
            depth_numpy = depth_numpy.squeeze()

        self.logger_test.debug("16bit depth/normal:")
        self.logger_test.debug(str(os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_rec_16bit.png")))
        
        # 写入文件
        cv2.imwrite(os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_{d_bpp_psnr}_rec_16bit.png"), depth_numpy)

        self.updateAvgMeter(avgMeter, rgb_p, rgb_m, rgb_bpp, depth_p, depth_m, depth_bpp, dec_time, enc_time)
        self.logger_test.info(
            f"Image[{i}:{img_name[0]}] | "
            f"rBpp loss: {rgb_bpp:.4f} | "
            f"dBpp loss: {depth_bpp:.4f} | "
            f"rPSNR: {rgb_p:.4f} | "
            f"dPSNR: {depth_p:.4f} | "
            f"rMS-SSIM: {rgb_m:.4f} | "
            f"dMS-SSIM: {depth_m:.4f} | "
            f"Encoding Latency: {enc_time:.4f} | "
            f"Decoding latency: {dec_time:.4f}"
        )

    def test_finish_log(self, avgMeter, rec_dir):
        self.logger_test.info(
            f"Epoch:[{self.epoch}] | "
            f"Avg rBpp: {avgMeter['avg_rgb_bpp'].avg:.7f} | "
            f"Avg dBpp: {avgMeter['avg_depth_bpp'].avg:.7f} | "
            f"Avg rPSNR: {avgMeter['avg_rgb_psnr'].avg:.7f} | "
            f"Avg dPSNR: {avgMeter['avg_depth_psnr'].avg:.7f} | "
            f"Avg rMS-SSIM: {avgMeter['avg_rgb_ms_ssim'].avg:.7f} | "
            f"Avg dMS-SSIM: {avgMeter['avg_depth_ms_ssim'].avg:.7f} | "
            f"Avg Encoding Latency: {avgMeter['avg_encode_time'].avg:.6f} | "
            f"Avg Decoding latency: {avgMeter['avg_deocde_time'].avg:.6f}"
        )

        self.write_test_img_name(os.path.join(rec_dir, "depth_rec"), os.path.join(rec_dir, "test_depth.txt"))
        self.write_test_img_name(os.path.join(rec_dir, "rgb_rec"), os.path.join(rec_dir, "test_rgb.txt"))

    def compress_one_image_united(self, x, stream_path, H, W, img_name):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = self.net.compress(x[0], x[1])
        torch.cuda.synchronize()
        end_time = time.time()
        shape = out["shape"]
        os.makedirs(stream_path[0], exist_ok=True)
        os.makedirs(stream_path[1], exist_ok=True)

        rgb_output = os.path.join(stream_path[0], img_name)
        with Path(rgb_output).open("wb") as f:
            write_uints(f, (H, W))
            write_body(f, shape, out["r_strings"])
        size = filesize(rgb_output)
        rgb_bpp = float(size) * 8 / (H * W)

        depth_output = os.path.join(stream_path[1], img_name)
        with Path(depth_output).open("wb") as f:
            write_uints(f, (H, W))
            write_body(f, shape, out["d_strings"])
        size = filesize(depth_output)
        depth_bpp = float(size) * 8 / (H * W)

        enc_time = end_time - start_time
        return rgb_bpp, depth_bpp, enc_time

    def decompress_one_image_united(self, stream_path, img_name, mode="reflect0"):
        rgb_output = os.path.join(stream_path[0], img_name)
        with Path(rgb_output).open("rb") as f:
            original_size = read_uints(f, 2)
            rgb_strings, shape = read_body(f)

        depth_output = os.path.join(stream_path[1], img_name)
        with Path(depth_output).open("rb") as f:
            original_size = read_uints(f, 2)
            depth_strings, shape = read_body(f)

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            out = self.net.decompress(rgb_strings, depth_strings, shape)
        torch.cuda.synchronize()
        end_time = time.time()
        dec_time = end_time - start_time
        rgb_x_hat = out["x_hat"]["r"]
        depth_x_hat = out["x_hat"]["d"]
        if mode.find("0") != -1:
            rgb_x_hat = crop0(rgb_x_hat, original_size)
            depth_x_hat = crop0(depth_x_hat, original_size)
        else:
            rgb_x_hat = crop1(rgb_x_hat, original_size)
            depth_x_hat = crop1(depth_x_hat, original_size)
        return rgb_x_hat, depth_x_hat, dec_time
