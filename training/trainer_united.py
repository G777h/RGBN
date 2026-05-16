import os
import sys

sys.path.append("..")  # cd xx/playground

import os
from pprint import pprint

# pyrefly: ignore [missing-import]
import torch
from utils.IOutils import saveImg
from utils.metrics import AverageMeter, compute_metrics
from utils.rd_loss import RateDistortionLossUnited

from .trainer import Trainer


class TrainerUnited(Trainer):
    def __init__(self, args, model_config) -> None:
        super().__init__(args, model_config)
        self.criterion = RateDistortionLossUnited(
            quality=args.quality,
            distortionLossForDepth=args.distortionLossForDepth,
            warmup_step=args.warmup_step,
            rgb_warmup_step=0,   # RGB 感知损失从第 1 步就启用（可通过参数扩展）
        )
    def train_backward_and_log(self, out_criterion, clip_max_norm, i, epoch, current_step):
        out_criterion["loss"].backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_max_norm)
        self.optimizer.step()
        
        # [Fix Issue-11] aux_loss 在主 loss backward 后立即计算，但 optimizer 已 step
        # 改为：aux_optimizer.zero_grad 在训练循环处已清空，直接调用 aux backward 即可
        # 这里对 entropy bottleneck 参数独立优化，不依赖主干纰
        self.aux_optimizer.zero_grad()
        aux_loss = self.net.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()

        current_step += 1
        
        if (current_step % 100 == 0 or self.debug) and self.rank == 0:
            self.tb_logger.add_scalar("[train]: loss", out_criterion["loss"].item(), current_step)
            self.tb_logger.add_scalar("[train]: rbpp_loss", out_criterion["r_bpp_loss"].item(), current_step)
            self.tb_logger.add_scalar("[train]: dbpp_loss", out_criterion["d_bpp_loss"].item(), current_step)
            self.tb_logger.add_scalar("[train]: rmse_loss", out_criterion["r_mse_loss"].item(), current_step)
            self.tb_logger.add_scalar("[train]: d_pure_mse", out_criterion["d_pure_mse"].item(), current_step)
            if "d_loss" in out_criterion:
                self.tb_logger.add_scalar("[train]: d_loss", out_criterion["d_loss"].item(), current_step)
            # 新增 RGB 感知损失监控
            if "r_perc_loss" in out_criterion:
                self.tb_logger.add_scalar("[train]: r_perc_loss", out_criterion["r_perc_loss"].item(), current_step)
                self.tb_logger.add_scalar("[train]: r_ssim_loss", out_criterion["r_ssim_loss"].item(), current_step)

        if (i % 100 == 0 or self.debug) and self.rank == 0:
            d_loss_str = f'{out_criterion["d_loss"].item():.4f}' if "d_loss" in out_criterion else "Warmup(MSE)"
            r_perc_str = f'{out_criterion["r_perc_loss"].item():.4f}' if "r_perc_loss" in out_criterion else "MSE"
            self.logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*self.train_dataloader.batch_size:5d}/{len(self.train_dataloader.dataset)}"
                f" ({100. * i / len(self.train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.4f} | '
                f'rMSE: {out_criterion["r_mse_loss"].item():.4f} | '
                f'dMSE: {out_criterion["d_pure_mse"].item():.4f} | '
                f'rBpp: {out_criterion["r_bpp_loss"].item():.4f} | '
                f'dBpp: {out_criterion["d_bpp_loss"].item():.4f} | '
                f'rPerc: {r_perc_str} | '
                f'dLoss(Perc): {d_loss_str} | '
                f"Aux: {aux_loss.item():.2f}"
            )
        return current_step


    def forward(self, d):
        rgb = d[0].to(self.device)
        depth = d[1].to(self.device)
        out_net = self.net(rgb, depth)
        out_criterion = self.criterion(out_net, rgb, depth)
        return out_criterion, out_net

    def train_one_epoch(self, epoch, current_step, clip_max_norm=1):
        self.net.train()
        for i, d in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()
            out_criterion, _ = self.forward(d)
            current_step = self.train_backward_and_log(out_criterion, clip_max_norm, i, epoch, current_step)
        return current_step

    def getAvgMeter(self):
        return {
            "loss": AverageMeter(),
            "r_bpp_loss": AverageMeter(),
            "d_bpp_loss": AverageMeter(),
            "r_mse_loss": AverageMeter(),
            "d_loss": AverageMeter(),
            "aux_loss": AverageMeter(),
            "r_psnr": AverageMeter(),
            "d_psnr": AverageMeter(),
            "r_ms_ssim": AverageMeter(),
            "d_ms_ssim": AverageMeter(),
        }

    def updateAvgMeter(self, avgMeter, out_criterion, rec, gt):
        avgMeter["r_bpp_loss"].update(out_criterion["r_bpp_loss"])
        avgMeter["d_bpp_loss"].update(out_criterion["d_bpp_loss"])
        avgMeter["loss"].update(out_criterion["loss"])
        avgMeter["r_mse_loss"].update(out_criterion["r_mse_loss"])
        # d_loss 仅在感知损失阶段存在，warmup 期间用 d_pure_mse 代替
        d_loss_val = out_criterion.get("d_loss", out_criterion.get("d_pure_mse", out_criterion["loss"]))
        avgMeter["d_loss"].update(d_loss_val)

        rec_r = rec["r"].clamp_(0, 1)
        p, m = compute_metrics(rec_r, gt[0])
        avgMeter["r_psnr"].update(p)
        avgMeter["r_ms_ssim"].update(m)
        rec_d = rec["d"].clamp_(0, 1)
        p, m = compute_metrics(rec_d, gt[1])
        avgMeter["d_psnr"].update(p)
        avgMeter["d_ms_ssim"].update(m)

    def validate_log(self, avgMeter, epoch):
        if self.rank == 0:
            self.tb_logger.add_scalar("{}".format("[val]: loss"), avgMeter["loss"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: rbpp_loss"), avgMeter["r_bpp_loss"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: dbpp_loss"), avgMeter["d_bpp_loss"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: rpsnr"), avgMeter["r_psnr"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: dpsnr"), avgMeter["d_psnr"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: rms-ssim"), avgMeter["r_ms_ssim"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: dms-ssim"), avgMeter["d_ms_ssim"].avg, epoch + 1)
            self.logger_val.info(
                f"Test epoch {epoch}: Average losses: "
                f"Loss: {avgMeter['loss'].avg:.4f} | "
                f"rMSE loss: {avgMeter['r_mse_loss'].avg:.4f} | "
                f"d loss: {avgMeter['d_loss'].avg:.4f} | "
                f"rBpp loss: {avgMeter['r_bpp_loss'].avg:.4f} | "
                f"dBpp loss: {avgMeter['d_bpp_loss'].avg:.4f} | "
                f"Aux loss: {avgMeter['aux_loss'].avg:.2f} | "
                f"rPSNR: {avgMeter['r_psnr'].avg:.4f} | "
                f"dPSNR: {avgMeter['d_psnr'].avg:.4f} | "
                f"rMS-SSIM: {avgMeter['r_ms_ssim'].avg:.4f} |"
                f"dMS-SSIM: {avgMeter['d_ms_ssim'].avg:.4f}\n"
            )
            self.tb_logger.add_scalar("{}".format("[val]: r_mse_loss"), avgMeter["r_mse_loss"].avg, epoch + 1)
            self.tb_logger.add_scalar("{}".format("[val]: d_loss"), avgMeter["d_loss"].avg, epoch + 1)

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.net.eval()
        device = next(self.net.parameters()).device
        save_dir = os.path.join(self.exp_dir_path, "val_images", "%03d" % (epoch + 1))
        os.makedirs(save_dir, exist_ok=True)

        avgMeter = self.getAvgMeter()
        for i, d in enumerate(self.val_dataloader):
            out_criterion, out_net = self.forward(d)
            self.updateAvgMeter(avgMeter, out_criterion, out_net["x_hat"], (d[0], d[1]))

            if (i % 20 == 1 or self.debug) and self.rank == 0:
                saveImg(out_net["x_hat"]["r"][0], os.path.join(save_dir, "%03d_rgb_rec.png" % i))
                saveImg(out_net["x_hat"]["d"][0], os.path.join(save_dir, "%03d_depth_rec.png" % i))
                saveImg(d[0][0], os.path.join(save_dir, "%03d_rgb_gt.png" % i))
                saveImg(d[1][0], os.path.join(save_dir, "%03d_depth_gt.png" % i))
        self.validate_log(avgMeter, epoch)
        return avgMeter["loss"].avg


if __name__ == "__main__":
    pass
