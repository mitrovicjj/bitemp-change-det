import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from torch.nn.modules.padding import ReplicationPad2d
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.local import LocalChangeDataset
from src.oscd import OSCDDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(ch, feature))
            ch = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous()
        targets = targets.contiguous()

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.psi(attn)
        return x * attn


class HybridDiffConcatFusion(nn.Module):
    """
    Kombinuje abs(f1-f2) i concat(f1,f2), pa vraća skip tensor iste širine
    kao originalni diff branch.
    """
    def __init__(self, ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)           # [B, ch, H, W]
        conc = torch.cat([f1, f2], dim=1)   # [B, 2ch, H, W]
        x = torch.cat([diff, conc], dim=1)  # [B, 3ch, H, W]
        return self.fuse(x)                 # [B, ch, H, W]
    
class SiamUnet_diff(nn.Module):
    """SiamUnet_diff segmentation network adapted for binary change detection."""

    def __init__(self, input_nbr, label_nbr=1, use_attention=True, use_hybrid=True):
        super(SiamUnet_diff, self).__init__()

        self.input_nbr = input_nbr
        self.use_attention = use_attention
        self.use_hybrid = use_hybrid

        # ---------------- Encoder ----------------
        self.conv11 = nn.Conv2d(input_nbr, 32, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.do11 = nn.Dropout2d(p=0.2)

        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.do12 = nn.Dropout2d(p=0.2)

        self.conv21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        self.do21 = nn.Dropout2d(p=0.2)

        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.do22 = nn.Dropout2d(p=0.2)

        self.conv31 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(128)
        self.do31 = nn.Dropout2d(p=0.2)

        self.conv32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(128)
        self.do32 = nn.Dropout2d(p=0.2)

        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(128)
        self.do33 = nn.Dropout2d(p=0.2)

        self.conv41 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(256)
        self.do41 = nn.Dropout2d(p=0.2)

        self.conv42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(256)
        self.do42 = nn.Dropout2d(p=0.2)

        self.conv43 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(256)
        self.do43 = nn.Dropout2d(p=0.2)

        # ---------------- Decoder ----------------
        self.upconv4 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv43d = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(256)
        self.do43d = nn.Dropout2d(p=0.2)

        self.conv42d = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(256)
        self.do42d = nn.Dropout2d(p=0.2)

        self.conv41d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(128)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv33d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(128)
        self.do33d = nn.Dropout2d(p=0.2)

        self.conv32d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(128)
        self.do32d = nn.Dropout2d(p=0.2)

        self.conv31d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(64)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv22d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(64)
        self.do22d = nn.Dropout2d(p=0.2)

        self.conv21d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(32)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, padding=1, stride=2, output_padding=1
        )

        self.conv12d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(32)
        self.do12d = nn.Dropout2d(p=0.2)

        self.conv11d = nn.ConvTranspose2d(32, label_nbr, kernel_size=3, padding=1)

        if self.use_attention:
            self.att4 = AttentionGate(gate_ch=256, skip_ch=256, inter_ch=128)
            self.att3 = AttentionGate(gate_ch=128, skip_ch=128, inter_ch=64)
            self.att2 = AttentionGate(gate_ch=64, skip_ch=64, inter_ch=32)
            self.att1 = AttentionGate(gate_ch=32, skip_ch=32, inter_ch=16)

        if self.use_hybrid:
            self.fuse4 = HybridDiffConcatFusion(256)
            self.fuse3 = HybridDiffConcatFusion(128)
            self.fuse2 = HybridDiffConcatFusion(64)
            self.fuse1 = HybridDiffConcatFusion(32)


    def encode(self, x):
        x1 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x2 = self.do12(F.relu(self.bn12(self.conv12(x1))))
        p1 = F.max_pool2d(x2, 2, 2)

        x3 = self.do21(F.relu(self.bn21(self.conv21(p1))))
        x4 = self.do22(F.relu(self.bn22(self.conv22(x3))))
        p2 = F.max_pool2d(x4, 2, 2)

        x5 = self.do31(F.relu(self.bn31(self.conv31(p2))))
        x6 = self.do32(F.relu(self.bn32(self.conv32(x5))))
        x7 = self.do33(F.relu(self.bn33(self.conv33(x6))))
        p3 = F.max_pool2d(x7, 2, 2)

        x8 = self.do41(F.relu(self.bn41(self.conv41(p3))))
        x9 = self.do42(F.relu(self.bn42(self.conv42(x8))))
        x10 = self.do43(F.relu(self.bn43(self.conv43(x9))))
        p4 = F.max_pool2d(x10, 2, 2)

        return x2, x4, x7, x10, p4


    def forward(self, x1, x2):
        # shared encoder (Siamese weight sharing)
        f1_1, f1_2, f1_3, f1_4, f1_p = self.encode(x1)
        f2_1, f2_2, f2_3, f2_4, f2_p = self.encode(x2)

        # ---------------- Decoder ----------------
        x = self.upconv4(f1_p - f2_p)

        if self.use_hybrid:
            skip4 = self.fuse4(f1_4, f2_4)
        else:
            skip4 = torch.abs(f1_4 - f2_4)

        if self.use_attention:
            skip4 = self.att4(x, skip4)

        x = torch.cat([x, skip4], dim=1)
        x = self.do43d(F.relu(self.bn43d(self.conv43d(x))))
        x = self.do42d(F.relu(self.bn42d(self.conv42d(x))))
        x = self.do41d(F.relu(self.bn41d(self.conv41d(x))))

        # level 3
        x = self.upconv3(x)

        if self.use_hybrid:
            skip3 = self.fuse3(f1_3, f2_3)
        else:
            skip3 = torch.abs(f1_3 - f2_3)

        if self.use_attention:
            skip3 = self.att3(x, skip3)

        x = torch.cat([x, skip3], dim=1)
        x = self.do33d(F.relu(self.bn33d(self.conv33d(x))))
        x = self.do32d(F.relu(self.bn32d(self.conv32d(x))))
        x = self.do31d(F.relu(self.bn31d(self.conv31d(x))))

        # level 2
        x = self.upconv2(x)

        if self.use_hybrid:
            skip2 = self.fuse2(f1_2, f2_2)
        else:
            skip2 = torch.abs(f1_2 - f2_2)

        if self.use_attention:
            skip2 = self.att2(x, skip2)

        x = torch.cat([x, skip2], dim=1)
        x = self.do22d(F.relu(self.bn22d(self.conv22d(x))))
        x = self.do21d(F.relu(self.bn21d(self.conv21d(x))))

        # level 1
        x = self.upconv1(x)

        if self.use_hybrid:
            skip1 = self.fuse1(f1_1, f2_1)
        else:
            skip1 = torch.abs(f1_1 - f2_1)

        if self.use_attention:
            skip1 = self.att1(x, skip1)

        x = torch.cat([x, skip1], dim=1)
        x = self.do12d(F.relu(self.bn12d(self.conv12d(x))))
        x = self.conv11d(x)

        return x

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss
    

class FocalLoss(nn.Module):
    """Binary focal loss computed directly from logits (numerically stable)."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(probs, self.alpha),
            torch.full_like(probs, 1 - self.alpha),
        )
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalDiceLoss(nn.Module):
    """Weighted combination of FocalLoss and soft Dice loss."""

    def __init__(self, focal_weight=0.3, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()
        return self.focal_weight * focal_loss + (1.0 - self.focal_weight) * dice_loss

# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_zeroshot_and_save(model, loader, device, save_dir, threshold=0.5):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0

    for t1, t2, mask in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)

        if getattr(model, "input_nbr", None) is not None:
            logits = model(t1, t2)
        else:
            x = torch.cat([t1, t2], dim=1)
            logits = model(x)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i in range(t1.size(0)):
            sample_idx += 1

            prob_np = probs[i].squeeze().cpu().numpy()
            pred_np = preds[i].squeeze().cpu().numpy()

            np.save(save_dir / f"sample_{sample_idx:04d}_prob.npy", prob_np)
            np.save(save_dir / f"sample_{sample_idx:04d}_pred.npy", pred_np)

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))

            t1_img = t1[i].detach().cpu()
            t2_img = t2[i].detach().cpu()

            if t1_img.shape[0] == 3:
                t1_img = t1_img.permute(1, 2, 0).numpy()
                t2_img = t2_img.permute(1, 2, 0).numpy()
                axes[0].imshow(np.clip(t1_img, 0, 1))
                axes[1].imshow(np.clip(t2_img, 0, 1))
            else:
                axes[0].imshow(t1_img[0].numpy(), cmap="gray")
                axes[1].imshow(t2_img[0].numpy(), cmap="gray")

            axes[0].set_title("T1")
            axes[1].set_title("T2")
            axes[2].imshow(prob_np, cmap="viridis")
            axes[2].set_title("Prob")
            axes[3].imshow(pred_np, cmap="gray")
            axes[3].set_title("Pred")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            fig.savefig(save_dir / f"sample_{sample_idx:04d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Zero-shot predictions saved to: {save_dir}")
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, max_vis=3):
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    total_precision = total_recall = total_f1 = total_pixel_acc = 0.0
    batches = 0

    all_preds = []
    all_targets = []
    vis_samples = []

    for t1, t2, mask in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device)

        if getattr(model, "input_nbr", None) is not None:
            logits = model(t1, t2)
        else:
            x = torch.cat([t1, t2], dim=1)
            logits = model(x)

        loss = criterion(logits, mask)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        tp = (preds * mask).sum(dim=(1, 2, 3))
        fp = (preds * (1 - mask)).sum(dim=(1, 2, 3))
        fn = ((1 - preds) * mask).sum(dim=(1, 2, 3))
        tn = ((1 - preds) * (1 - mask)).sum(dim=(1, 2, 3))

        iou = ((tp + 1e-6) / (tp + fp + fn + 1e-6)).mean().item()
        dice = ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item()
        precision = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
        recall = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
        f1 = ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item()
        pixel_acc = ((tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)).mean().item()

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_pixel_acc += pixel_acc
        batches += 1

        all_preds.append(preds.cpu().numpy().astype(np.uint8).ravel())
        all_targets.append(mask.cpu().numpy().astype(np.uint8).ravel())

        if max_vis > 0 and len(vis_samples) < max_vis:
            for i in range(min(t1.size(0), max_vis - len(vis_samples))):
                vis_samples.append(
                    {
                        "t1": t1[i].cpu(),
                        "t2": t2[i].cpu(),
                        "mask": mask[i].cpu(),
                        "pred": preds[i].cpu(),
                        "prob": probs[i].cpu(),
                    }
                )

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.uint8)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.uint8)

    tp_g = int(((y_pred == 1) & (y_true == 1)).sum())
    fp_g = int(((y_pred == 1) & (y_true == 0)).sum())
    fn_g = int(((y_pred == 0) & (y_true == 1)).sum())
    tn_g = int(((y_pred == 0) & (y_true == 0)).sum())

    dice_global = (2 * tp_g + 1e-6) / (2 * tp_g + fp_g + fn_g + 1e-6)
    iou_global = (tp_g + 1e-6) / (tp_g + fp_g + fn_g + 1e-6)

    n = max(batches, 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "precision": total_precision / n,
        "recall": total_recall / n,
        "f1": total_f1 / n,
        "pixel_accuracy": total_pixel_acc / n,
        "confusion": {"tp": tp_g, "fp": fp_g, "fn": fn_g, "tn": tn_g},
        "vis_samples": vis_samples,
        "dice_global": dice_global,
        "iou_global": iou_global
    }


# ---------------------------------------------------------------------------
# Threshold sweep helper
# ---------------------------------------------------------------------------
def threshold_sweep(model, loader, criterion, device, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    results = []

    for thr in thresholds:
        metrics = evaluate(
            model,
            loader,
            criterion,
            device,
            threshold=thr,
            max_vis=0
        )

        results.append({
            "threshold": thr,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "pixel_accuracy": metrics["pixel_accuracy"],
        })

    best = max(results, key=lambda x: (x["dice"] + x["iou"]) / 2)

    return results, best


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.should_stop = False

    def step(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for t1, t2, mask in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device)


        if isinstance(model, SiamUnet_diff):
            logits = model(t1, t2)
        else:
            x = torch.cat([t1, t2], dim=1)
            logits = model(x)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_training_curves(history, out_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_iou"], label="val_iou")
    axes[1].plot(epochs, history["val_dice"], label="val_dice")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(conf, out_path):
    cm = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=["No change", "Change"])
    ax.set_yticks([0, 1], labels=["No change", "Change"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prediction_visuals(vis_samples, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for idx, sample in enumerate(vis_samples, start=1):
        t1 = sample["t1"].detach().cpu()
        t2 = sample["t2"].detach().cpu()
        mask = sample["mask"].squeeze(0).detach().cpu().numpy()
        pred = sample["pred"].squeeze(0).detach().cpu().numpy()
        prob = sample["prob"].squeeze(0).detach().cpu().numpy()

        if t1.shape[0] == 3:
            t1 = t1.permute(1, 2, 0).numpy()
            t2 = t2.permute(1, 2, 0).numpy()

            # de-normalizacija iz ImageNet prostora nazad u [0, 1]
            t1 = t1 * std + mean
            t2 = t2 * std + mean

            t1 = np.clip(t1, 0, 1)
            t2 = np.clip(t2, 0, 1)
        else:
            t1 = t1.squeeze(0).numpy()
            t2 = t2.squeeze(0).numpy()

            # za non-RGB fallback
            t1 = np.clip(t1, 0, 1)
            t2 = np.clip(t2, 0, 1)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        if t1.ndim == 3:
            axes[0].imshow(t1)
            axes[1].imshow(t2)
        else:
            axes[0].imshow(t1, cmap="gray")
            axes[1].imshow(t2, cmap="gray")

        axes[0].set_title("T1")
        axes[1].set_title("T2")
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("GT Mask")
        axes[3].imshow(prob, cmap="viridis")
        axes[3].set_title("Pred Prob")
        axes[4].imshow(pred, cmap="gray")
        axes[4].set_title("Pred Mask")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        fig.savefig(out_dir / f"prediction_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, device):

    if args.model_name == "unet":
        model = UNet(
            in_channels=6,
            out_channels=1,
            features=(32, 64, 128, 256)
        ).to(device)

    elif args.model_name == "siamdiff":
        model = SiamUnet_diff(
            input_nbr=3,
            label_nbr=1,
            use_attention=args.use_attention,
            use_hybrid=args.use_hybrid,
        ).to(device)

    elif args.model_name == "unet_resnet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=6,
            classes=1,
            activation=None
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    return model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"DEBUG mode = {args.mode}")
    print(f"DEBUG local_val_dir = {args.local_val_dir}")
    print(f"DEBUG checkpoint_path = {args.checkpoint_path}")

    out_dir = Path(args.output_dir)
    artifact_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    if args.mode == "finetune":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path je obavezan za finetune")
        model = load_checkpoint(model, args.checkpoint_path, device)
    elif args.mode == "zeroshot":
        print("DEBUG ENTERED ZEROSHOT BRANCH")
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path je obavezan za zeroshot")
        if args.local_val_dir is None:
            raise ValueError("--local-val-dir je obavezan za zeroshot")

        model = load_checkpoint(model, args.checkpoint_path, device)

        eval_ds = LocalChangeDataset(
            patches_dir=args.local_val_dir,
            normalize="scale_10000",
            apply_imagenet_norm=True,
            augment="none",
            patch_size=args.patch_size,
            crop_mode="none",
            split="val",
            num_crops_per_image=1,
        )

        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        criterion = DiceLoss()

        metrics = evaluate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            max_vis=3,
        )

        print(
            f"Zero-shot | "
            f"loss={metrics['loss']:.4f} | "
            f"iou={metrics['iou']:.4f} | "
            f"dice={metrics['dice']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f}"
        )

        save_prediction_visuals(
            metrics["vis_samples"],
            Path(args.save_preds_dir)
        )

        save_confusion_matrix(
            metrics["confusion"],
            artifact_dir / "zeroshot_confusion_matrix.png"
        )

        return  # ← izlaz iz main(), ne ulazi u trening
    print("DEBUG ENTERED TRAIN/FINETUNE BRANCH")
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    if args.local_train_dir is not None and args.local_val_dir is not None:
        train_ds = LocalChangeDataset(
            patches_dir=args.local_train_dir,
            normalize="scale_10000",
            apply_imagenet_norm=True,
            augment=args.augment,
            patch_size=args.patch_size,
            crop_mode="none",
            split="train",
            num_crops_per_image=1,
        )

        val_ds = LocalChangeDataset(
            patches_dir=args.local_val_dir,
            normalize="scale_10000",
            apply_imagenet_norm=True,
            augment="none",
            patch_size=args.patch_size,
            crop_mode="none",
            split="val",
            num_crops_per_image=1,
        )
    else:
        train_ds = OSCDDataset(
            split="train",
            patch_size=args.patch_size,
            crop_mode="random_crop",
            augment=args.augment,
            num_crops_per_image=20,
        )

        val_ds = OSCDDataset(
            split="test",
            patch_size=args.patch_size,
            crop_mode="center_crop",
            augment="none",
            num_crops_per_image=1,
        )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=g,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    pos_count = 0
    total_count = 0

    for i in range(len(train_ds)):
        _, _, mask = train_ds[i]
        pos_count += mask.sum().item()
        total_count += mask.numel()

    neg_count = total_count - pos_count

    eps = 1e-6
    pos_weight = np.sqrt((neg_count + eps) / (pos_count + eps))
    pos_weight = float(np.clip(pos_weight, 1.0, 10.0))

    print(f"Class balance: pos={pos_count}, neg={neg_count}, pos_weight={pos_weight:.2f}")


    pos_weight_tensor = torch.tensor(pos_weight, device=device)


    if args.loss_name == "dice":
        criterion = DiceLoss()
    elif args.loss_name == "bce_dice":
        criterion = BCEDiceLoss(
            bce_weight=args.bce_weight,
            pos_weight=pos_weight_tensor
        )
    elif args.loss_name == "focal_dice":
        criterion = FocalDiceLoss(
            focal_weight=args.focal_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    else:
        raise ValueError(
            f"Unsupported loss_name='{args.loss_name}'. Use 'dice', 'bce_dice' or 'focal_dice'."
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_path = out_dir / f"best_{args.model_name}_oscd.pt"
    last_path = out_dir / f"last_{args.model_name}_oscd.pt"

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_val_loss = float("inf")
    best_val_iou = 0.0
    best_val_dice = 0.0
    best_epoch = 0

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags(
            {
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "device": "cpu",
                "baseline": "true" if args.run_name.startswith("baseline") else "false",
            }
        )

        mlflow.log_params(
            {
                "patch_size": args.patch_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "train_mode": args.train_mode,
                "val_mode": args.val_mode,
                "augment": args.augment,
                "loss_name": args.loss_name,
                "bce_weight": args.bce_weight,
                "focal_weight": args.focal_weight,
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "seed": args.seed,
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "use_attention": args.use_attention,
                "use_hybrid": args.use_hybrid,
            }
        )

        early_stopping = EarlyStopping(patience=7, min_delta=0.002)

        epoch = 0
        best_score = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device, max_vis=0)
            scheduler.step(val_metrics["dice"])
            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"lr={lr_now:.6f}"
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_iou"].append(val_metrics["iou"])
            history["val_dice"].append(val_metrics["dice"])

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_iou": val_metrics["iou"],
                    "val_dice": val_metrics["dice"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                    "learning_rate": lr_now,
                },
                step=epoch,
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "args": vars(args),
            }
            torch.save(checkpoint, last_path)

            score = (val_metrics["dice"] + val_metrics["iou"]) / 2
            if score > best_score:
                best_score = score
                best_val_loss = val_metrics["loss"]
                best_val_iou = val_metrics["iou"]
                best_val_dice = val_metrics["dice"]
                best_epoch = epoch
                torch.save(checkpoint, best_path)
                print(f"  => Saved new best model (epoch {epoch})")

            if early_stopping.step(-val_metrics["dice"], epoch):
                print(
                    f"Early stopping at epoch {epoch:03d}. "
                    f"Best epoch: {early_stopping.best_epoch:03d}, "
                    f"val_negative_dice={early_stopping.best_loss:.4f}"
                )
                break

        mlflow.log_metrics(
            {
                "early_stopped_epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_iou": best_val_iou,
                "best_val_dice": best_val_dice,
                "best_epoch": best_epoch,
            }
        )

        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        best_threshold = 0.5

        final_eval = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=best_threshold,
            max_vis=3,
        )

        curves_path = artifact_dir / "training_curves.png"
        confusion_path = artifact_dir / "confusion_matrix.png"
        pred_dir = artifact_dir / "predictions"

        save_training_curves(history, curves_path)
        save_confusion_matrix(final_eval["confusion"], confusion_path)
        save_prediction_visuals(final_eval["vis_samples"], pred_dir)

        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(last_path))
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifacts(str(pred_dir), artifact_path="predictions")
        # mlflow.pytorch.log_model(model, artifact_path="model")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train U-Net or SiamDiff or pretrained resnet encoder on OSCD RGB with MLflow logging (CPU only)"
    )

    # Data
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--train-mode", type=str, default="random_crop")
    parser.add_argument("--val-mode", type=str, default="center_crop")
    parser.add_argument(
        "--augment",
        type=str,
        default="none",
        choices=["none", "flip", "flip_rot90", "strong"],
        help="Augmentation mode: none | flip | flip_rot90 | strong (flip_rot90 + photometric)",
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "zeroshot", "finetune"])

    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Putanja do pretrained/best checkpointa za zero-shot ili finetune.")

    parser.add_argument("--local-data-dir", type=str, default=None,
                        help="Root direktorijum lokalnog dataseta sa t1/t2/labels.")

    parser.add_argument("--local-train-dir", type=str, default=None,
                        help="Putanja do lokalnog TRAIN foldera")

    parser.add_argument("--local-val-dir", type=str, default=None,
                        help="Putanja do lokalnog VAL foldera")

    parser.add_argument("--save-preds-dir", type=str, default="predictions_local",
                        help="Gdje da sačuva zero-shot predikcije.")

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold za binarnu masku.")

    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--use-hybrid", action="store_true")

    # Loss
    parser.add_argument(
        "--loss-name",
        type=str,
        default="bce_dice",
        choices=["dice", "bce_dice", "focal_dice"],
        help="Loss function: bce_dice | focal_dice | dice",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="BCE share in BCEDiceLoss (only used when --loss-name=bce_dice)",
    )
    parser.add_argument(
        "--focal-weight",
        type=float,
        default=0.3,
        help="Focal share in FocalDiceLoss (only used when --loss-name=focal_dice)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal alpha (class-balance weight)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal gamma (focusing parameter)",
    )

    # Threshold sweep
    parser.add_argument(
        "--do-threshold-sweep",
        action="store_true",
        help="Run threshold sweep on best checkpoint after training",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Decision thresholds to evaluate during sweep",
    )

    # MLflow / output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default="oscd_change_detection")
    parser.add_argument("--run-name", type=str, default="baseline_unet_cpu")
    parser.add_argument(
        "--model-name",
        type=str,
        default="unet",
        choices=["unet", "siamdiff", "unet_resnet"]
    )
    parser.add_argument("--dataset-name", type=str, default="blanchon/OSCD_RGB")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="file:./mlruns")

    args = parser.parse_args()
    main(args)