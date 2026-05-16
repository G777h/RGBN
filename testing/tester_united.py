import os
import time
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from dataset.testDataset import ImageFolderUnited
from dataset.utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.IOutils import *
from utils.metrics import AverageMeter

from .tester_single import TesterSingle
import lpips
from DISTS_pytorch import DISTS
from pytorch_msssim import ms_ssim

# ==================== 新增：Mask 处理与 Masked 指标计算函数 ====================
def read_mask_tensor(filepath, target_shape, device):
    
    if not os.path.exists(filepath):
        print(f"[Warning] Mask not found: {filepath}. Using full mask.")
        return torch.ones(target_shape, device=device)

    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    tensor = torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

def compute_masked_psnr(pred, target, mask):
    
    diff = (pred - target) * mask
    valid_pixels = mask.sum() * pred.size(1) # 有效像素数 * 通道数
    
    if valid_pixels == 0:
        return 0.0
        
    mse = torch.sum(diff ** 2) / valid_pixels
    if mse == 0:
        return 100.0
    return -10 * math.log10(mse)

def compute_masked_mae(pred, target, mask):
    
    pred_vec = pred * 2.0 - 1.0
    target_vec = target * 2.0 - 1.0

    pred_norm = F.normalize(pred_vec, p=2, dim=1, eps=1e-8)
    target_norm = F.normalize(target_vec, p=2, dim=1, eps=1e-8)

    cos_sim = torch.sum(pred_norm * target_norm, dim=1, keepdim=True)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angular_error_deg = torch.rad2deg(torch.acos(cos_sim))

    valid_errors = angular_error_deg[mask.bool().expand_as(angular_error_deg)]
    if valid_errors.numel() == 0:
        return 0.0
    return torch.mean(valid_errors).item()
# =========================================================================

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
            "avg_rgb_lpips": AverageMeter(),
            "avg_rgb_dists": AverageMeter(),
            "avg_rgb_bpp": AverageMeter(),
            "avg_depth_psnr": AverageMeter(),
            "avg_depth_ms_ssim": AverageMeter(),
            "avg_depth_lpips": AverageMeter(),
            "avg_depth_dists": AverageMeter(),
            "avg_depth_mae": AverageMeter(),  # 新增：法线图专用的 MAE
            "avg_depth_bpp": AverageMeter(),
            "avg_deocde_time": AverageMeter(),
            "avg_encode_time": AverageMeter(),
        }

    def updateAvgMeter(self, avgMeter, rgb_p, rgb_m, rgb_l, rgb_d, rgb_bpp, 
                       depth_p, depth_m, depth_l, depth_d, depth_mae, depth_bpp, dec_time, enc_time):
        avgMeter["avg_rgb_psnr"].update(rgb_p)
        avgMeter["avg_rgb_ms_ssim"].update(rgb_m)
        avgMeter["avg_rgb_lpips"].update(rgb_l)
        avgMeter["avg_rgb_dists"].update(rgb_d)
        avgMeter["avg_rgb_bpp"].update(rgb_bpp)
        
        avgMeter["avg_depth_psnr"].update(depth_p)
        avgMeter["avg_depth_ms_ssim"].update(depth_m)
        avgMeter["avg_depth_lpips"].update(depth_l)
        avgMeter["avg_depth_dists"].update(depth_d)
        avgMeter["avg_depth_mae"].update(depth_mae)  # 更新 MAE
        avgMeter["avg_depth_bpp"].update(depth_bpp)
        
        avgMeter["avg_deocde_time"].update(dec_time)
        avgMeter["avg_encode_time"].update(enc_time)

    @torch.no_grad()
    def test_model(self, padding_mode="reflect0", padding=True):
        self.net.eval()
        
        self.logger_test.info("Loading LPIPS and DISTS models...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device).eval()
        self.dists_fn = DISTS().to(self.device).eval()
        
        avgMeter = self.getAvgMeter()
        rec_dir = self.get_rec_dir(padding=padding, padding_mode=padding_mode)
        
        # 设定 Mask 存放的绝对路径
        mask_root = "/data1/Qihao_data/data/pswild/mask/"

        for i, (rgb, depth, rgb_img_name, depth_img_name) in enumerate(self.test_dataloader):
            B, C, H, W = rgb.shape

            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            
            # --- 读取对应的 Mask ---
            mask_name = f"{rgb_img_name[0]}_mask.png"  
            mask_file = os.path.join(mask_root, mask_name)
            mask = read_mask_tensor(mask_file, (1, 1, H, W), self.device)

            rgb_pad = pad(rgb, padding_mode)
            depth_pad = pad(depth, padding_mode)
            rgb_stream_path = os.path.join(rec_dir, "depth_bin")
            depth_stream_path = os.path.join(rec_dir, "rgb_bin")
            
            rgb_bpp, depth_bpp, enc_time = self.compress_one_image_united(
                x=(rgb_pad, depth_pad),
                stream_path=(rgb_stream_path, depth_stream_path),
                H=H, W=W,
                img_name=rgb_img_name[0],
            )
            
            rgb_x_hat, depth_x_hat, dec_time = self.decompress_one_image_united(
                stream_path=(rgb_stream_path, depth_stream_path), img_name=rgb_img_name[0], mode=padding_mode
            )
            
            self.test_save_and_log_perimg(
                i, rgb_x_hat, depth_x_hat, rgb, depth, mask,  # 传入 Mask
                rec_dir, rgb_img_name, avgMeter,
                rgb_bpp, depth_bpp, dec_time, enc_time,
            )
            
        self.test_finish_log(avgMeter, rec_dir)

    def test_save_and_log_perimg(
        self, i, rgb_x_hat, depth_x_hat, rgb, depth, mask, rec_dir, img_name, avgMeter, rgb_bpp, depth_bpp, dec_time, enc_time
    ):
        # --- 1. 使用 Mask 将背景清零 ---
        rgb_masked = rgb * mask
        rgb_x_hat_masked = rgb_x_hat * mask
        depth_masked = depth * mask
        depth_x_hat_masked = depth_x_hat * mask

        # --- 2. 计算 Masked PSNR 和 MAE ---
        rgb_p = compute_masked_psnr(rgb_x_hat_masked, rgb_masked, mask)
        depth_p = compute_masked_psnr(depth_x_hat_masked, depth_masked, mask)
        depth_mae = compute_masked_mae(depth_x_hat_masked, depth_masked, mask)

        # --- 3. 计算黑边化 MS-SSIM ---
        rgb_m = ms_ssim(rgb_x_hat_masked, rgb_masked, data_range=1.0).item()
        depth_m = ms_ssim(depth_x_hat_masked, depth_masked, data_range=1.0).item()
        
        # --- 4. 计算黑边化 LPIPS 和 DISTS ---
        with torch.no_grad():
            rgb_l = self.lpips_fn((rgb_x_hat_masked * 2 - 1), (rgb_masked * 2 - 1)).item()
            rgb_d = self.dists_fn(rgb_x_hat_masked, rgb_masked).item()

            d_x_hat_3c = depth_x_hat_masked.expand(-1, 3, -1, -1) if depth_x_hat_masked.size(1) == 1 else depth_x_hat_masked
            depth_3c = depth_masked.expand(-1, 3, -1, -1) if depth_masked.size(1) == 1 else depth_masked
            
            depth_l = self.lpips_fn((d_x_hat_3c * 2 - 1), (depth_3c * 2 - 1)).item()
            depth_d = self.dists_fn(d_x_hat_3c, depth_3c).item()
        
        r_bpp_psnr = f"{rgb_bpp:.4f}_{rgb_p:.4f}_"
        d_bpp_psnr = f"{depth_bpp:.4f}_{depth_p:.4f}_"

        # 注意：保存图片时，保存的是完整的无黑边重建图，方便论文展示对比
        saveImg(rgb_x_hat, os.path.join(rec_dir, "rgb_rec", f"{img_name[0]}_{r_bpp_psnr}_rec.png"))
        saveImg(depth_x_hat, os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_{d_bpp_psnr}_rec_8bit.png"))

        if rec_dir.find("sun") != -1:
            depth_16bit = depth_x_hat * 100000
        else:
            depth_16bit = depth_x_hat * 65535.0  
            
        depth_numpy = depth_16bit[0].cpu().numpy()
        depth_numpy = np.clip(depth_numpy, 0, 65535).astype(np.uint16)
        
        if depth_numpy.ndim == 3 and depth_numpy.shape[0] == 3:
            depth_numpy = np.transpose(depth_numpy, (1, 2, 0))
            depth_numpy = cv2.cvtColor(depth_numpy, cv2.COLOR_RGB2BGR)
        elif depth_numpy.ndim == 3 and depth_numpy.shape[0] == 1:
            depth_numpy = depth_numpy.squeeze()

        self.logger_test.debug("16bit depth/normal:")
        self.logger_test.debug(str(os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_rec_16bit.png")))
        cv2.imwrite(os.path.join(rec_dir, "depth_rec", f"{img_name[0]}_{d_bpp_psnr}_rec_16bit.png"), depth_numpy)

        # 传递包含 depth_mae 的更新指标
        self.updateAvgMeter(avgMeter, rgb_p, rgb_m, rgb_l, rgb_d, rgb_bpp, 
                            depth_p, depth_m, depth_l, depth_d, depth_mae, depth_bpp, dec_time, enc_time)
        
        self.logger_test.info(
            f"Image[{i}:{img_name[0]}] | "
            f"rBpp: {rgb_bpp:.4f} | dBpp: {depth_bpp:.4f} | "
            f"rPSNR: {rgb_p:.4f} | dPSNR: {depth_p:.4f} | "
            f"rMS-SSIM: {rgb_m:.4f} | dMS-SSIM: {depth_m:.4f} | "
            f"rLPIPS: {rgb_l:.4f} | dLPIPS: {depth_l:.4f} | " 
            f"rDISTS: {rgb_d:.4f} | dDISTS: {depth_d:.4f} | "
            f"dMAE: {depth_mae:.4f}"
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
            f"Avg rLPIPS: {avgMeter['avg_rgb_lpips'].avg:.7f} | "   
            f"Avg dLPIPS: {avgMeter['avg_depth_lpips'].avg:.7f} | " 
            f"Avg rDISTS: {avgMeter['avg_rgb_dists'].avg:.7f} | "   
            f"Avg dDISTS: {avgMeter['avg_depth_dists'].avg:.7f} | "
            f"Avg dMAE: {avgMeter['avg_depth_mae'].avg:.4f} | "     # 最终打印均值 MAE
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
