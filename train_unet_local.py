import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset

from src.local import LocalChangeDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    def __init__(self, in_channels=6, out_channels=1, features=(32, 64, 128, 256)):
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
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
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
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for t1, t2, mask in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device)
        x = torch.cat([t1, t2], dim=1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, max_vis=4):
    model.eval()
    total_loss = 0.0
    batches = 0
    all_preds = []
    all_targets = []
    vis_samples = []

    for t1, t2, mask in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device)
        x = torch.cat([t1, t2], dim=1)
        logits = model(x)
        loss = criterion(logits, mask)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        total_loss += loss.item()
        batches += 1
        all_preds.append(preds.cpu().numpy().astype(np.uint8).ravel())
        all_targets.append(mask.cpu().numpy().astype(np.uint8).ravel())

        if len(vis_samples) < max_vis:
            free_slots = max_vis - len(vis_samples)
            for i in range(min(t1.size(0), free_slots)):
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

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    pixel_accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    smooth = 1e-6
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    return {
        "loss": total_loss / max(batches, 1),
        "iou": float(iou),
        "f1": float(f1),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_accuracy),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "vis_samples": vis_samples,
    }


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
    axes[1].plot(epochs, history["val_f1"], label="val_f1")
    axes[1].plot(epochs, history["val_dice"], label="val_dice")
    axes[1].plot(epochs, history["val_iou"], label="val_iou")
    axes[1].plot(epochs, history["val_precision"], label="val_precision", alpha=0.7)
    axes[1].plot(epochs, history["val_recall"], label="val_recall", alpha=0.7)
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
    for idx, sample in enumerate(vis_samples, start=1):
        t1 = sample["t1"].permute(1, 2, 0).numpy()
        t2 = sample["t2"].permute(1, 2, 0).numpy()
        mask = sample["mask"].squeeze(0).numpy()
        pred = sample["pred"].squeeze(0).numpy()
        prob = sample["prob"].squeeze(0).numpy()
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(np.clip(t1, 0, 1))
        axes[0].set_title("T1")
        axes[1].imshow(np.clip(t2, 0, 1))
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


def save_history_csv(history, out_path):
    fieldnames = list(history.keys())
    rows = zip(*[history[k] for k in fieldnames])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(row)


def save_run_summary(summary_dict, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in summary_dict.items():
            f.write(f"{k}: {v}\n")


def make_train_val_subsets(args):
    train_base = LocalChangeDataset(
        patches_dir=args.patches_dir,
        patch_size=args.patch_size,
        crop_mode=args.train_mode,
        augment=args.augment,
        split="train",
        normalize=args.normalize,
    )
    n = len(train_base)
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    val_size = max(1, int(n * args.val_split))
    train_size = n - val_size

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    val_base = LocalChangeDataset(
        patches_dir=args.patches_dir,
        patch_size=args.patch_size,
        crop_mode=args.val_mode,
        augment="none",
        split="train",
        normalize=args.normalize,
    )

    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)
    return train_ds, val_ds, train_size, val_size


def main(args):
    set_seed(args.seed)
    device = torch.device('cpu')
    print(f'Using device: {device}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / 'artifacts'
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    if args.mode == 'zeroshot':
        val_ds = LocalChangeDataset(
            patches_dir=args.patches_dir,
            patch_size=args.patch_size,
            crop_mode=args.val_mode,
            augment='none',
            split='val',
            normalize=args.normalize,
        )
        train_ds = None
        train_loader = None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        train_size = 0
        val_size = len(val_ds)
    else:
        full_ds = LocalChangeDataset(
            patches_dir=args.patches_dir,
            patch_size=args.patch_size,
            crop_mode=args.train_mode,
            augment='none',
            split='train',
            normalize=args.normalize,
        )
        n = len(full_ds)
        indices = np.arange(n)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)
        val_size = max(1, int(n * args.val_split))
        train_size = n - val_size
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:].tolist()

        train_base = LocalChangeDataset(
            patches_dir=args.patches_dir,
            patch_size=args.patch_size,
            crop_mode=args.train_mode,
            augment=args.augment,
            split='train',
            normalize=args.normalize,
        )
        val_base = LocalChangeDataset(
            patches_dir=args.patches_dir,
            patch_size=args.patch_size,
            crop_mode=args.val_mode,
            augment='none',
            split='train',
            normalize=args.normalize,
        )
        train_ds = Subset(train_base, train_indices)
        val_ds = Subset(val_base, val_indices)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get('args', {})
    features = ckpt_args.get('features', '32,64,128,256')
    if isinstance(features, str):
        features = tuple(int(x) for x in features.split(','))
    else:
        features = tuple(features)
    model = UNet(in_channels=6, out_channels=1, features=features).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_loss_path = out_dir / ('best_local_eval_by_loss.pt' if args.mode == 'zeroshot' else 'best_finetuned_local_by_loss.pt')
    last_path = out_dir / ('last_local_eval.pt' if args.mode == 'zeroshot' else 'last_finetuned_local.pt')

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_pixel_accuracy': [],
        'learning_rate': [],
    }

    best_val_loss = float('inf')
    best_epoch_by_loss = 0
    best_val_f1 = 0.0
    best_val_iou_at_best_f1 = 0.0
    best_val_dice_at_best_f1 = 0.0
    best_val_precision_at_best_f1 = 0.0
    best_val_recall_at_best_f1 = 0.0
    best_epoch_by_f1 = 0

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({
            'model_name': args.model_name,
            'source_dataset': 'OSCD_RGB',
            'target_dataset': args.dataset_name,
            'mode': args.mode,
            'device': 'cpu',
            'task': 'binary_change_detection',
            'input_type': 'bitemporal_rgb',
            'selection_rule_best_checkpoint': 'lowest_val_loss',
            'report_metric_primary': 'val_f1',
        })
        mlflow.log_params({
            'patch_size': args.patch_size,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'train_mode': args.train_mode,
            'val_mode': args.val_mode,
            'augment': args.augment,
            'bce_weight': args.bce_weight,
            'seed': args.seed,
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'num_workers': args.num_workers,
            'checkpoint': args.checkpoint,
            'mode': args.mode,
            'normalize': args.normalize,
            'val_split': args.val_split,
            'in_channels': 6,
            'out_channels': 1,
            'features': ','.join(map(str, features)),
            'scheduler': 'ReduceLROnPlateau',
            'early_stopping_patience': args.early_stopping_patience,
            'early_stopping_min_delta': args.early_stopping_min_delta,
            'threshold': args.threshold,
            'train_size': train_size,
            'val_size': val_size,
        })

        if args.mode == 'zeroshot':
            metrics = evaluate(model, val_loader, criterion, device, threshold=args.threshold, max_vis=args.num_vis_samples)
            history['epoch'].append(0)
            history['train_loss'].append(0.0)
            history['val_loss'].append(metrics['loss'])
            history['val_iou'].append(metrics['iou'])
            history['val_dice'].append(metrics['dice'])
            history['val_f1'].append(metrics['f1'])
            history['val_precision'].append(metrics['precision'])
            history['val_recall'].append(metrics['recall'])
            history['val_pixel_accuracy'].append(metrics['pixel_accuracy'])
            history['learning_rate'].append(0.0)
            mlflow.log_metrics({
                'train_loss': 0.0,
                'val_loss': metrics['loss'],
                'val_iou': metrics['iou'],
                'val_dice': metrics['dice'],
                'val_f1': metrics['f1'],
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall'],
                'val_pixel_accuracy': metrics['pixel_accuracy'],
                'learning_rate': 0.0,
                'best_val_loss': metrics['loss'],
                'best_epoch_by_loss': 0,
                'best_val_f1': metrics['f1'],
                'best_val_iou_at_best_f1': metrics['iou'],
                'best_val_dice_at_best_f1': metrics['dice'],
                'best_val_precision_at_best_f1': metrics['precision'],
                'best_val_recall_at_best_f1': metrics['recall'],
                'best_epoch_by_f1': 0,
            })
            best_val_loss = metrics['loss']
            best_epoch_by_loss = 0
            best_val_f1 = metrics['f1']
            best_val_iou_at_best_f1 = metrics['iou']
            best_val_dice_at_best_f1 = metrics['dice']
            best_val_precision_at_best_f1 = metrics['precision']
            best_val_recall_at_best_f1 = metrics['recall']
            best_epoch_by_f1 = 0
            zero_ckpt = {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": None,
                "val_loss": metrics["loss"],
                "val_iou": metrics["iou"],
                "val_dice": metrics["dice"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "args": vars(args),
            }
            torch.save(zero_ckpt, best_loss_path)
            torch.save(zero_ckpt, last_path)
            final_eval = metrics
        else:
            early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)
            epoch = 0
            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_metrics = evaluate(model, val_loader, criterion, device, threshold=args.threshold, max_vis=args.num_vis_samples)
                scheduler.step(val_metrics['loss'])
                lr_now = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | val_iou={val_metrics['iou']:.4f} | val_dice={val_metrics['dice']:.4f} | val_precision={val_metrics['precision']:.4f} | val_recall={val_metrics['recall']:.4f} | lr={lr_now:.6f}")
                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_metrics['loss'])
                history['val_iou'].append(val_metrics['iou'])
                history['val_dice'].append(val_metrics['dice'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_precision'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_pixel_accuracy'].append(val_metrics['pixel_accuracy'])
                history['learning_rate'].append(lr_now)
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_iou': val_metrics['iou'],
                    'val_dice': val_metrics['dice'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_pixel_accuracy': val_metrics['pixel_accuracy'],
                    'learning_rate': lr_now,
                }, step=epoch)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_iou': val_metrics['iou'],
                    'val_dice': val_metrics['dice'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'args': vars(args),
                }
                torch.save(checkpoint, last_path)
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_epoch_by_loss = epoch
                    torch.save(checkpoint, best_loss_path)
                    print(f'Saved new best-loss model to {best_loss_path}')
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    best_val_iou_at_best_f1 = val_metrics['iou']
                    best_val_dice_at_best_f1 = val_metrics['dice']
                    best_val_precision_at_best_f1 = val_metrics['precision']
                    best_val_recall_at_best_f1 = val_metrics['recall']
                    best_epoch_by_f1 = epoch
                if early_stopping.step(val_metrics['loss'], epoch):
                    print(f"Early stopping at epoch {epoch:03d}. Best loss epoch was {early_stopping.best_epoch:03d} with val_loss={early_stopping.best_loss:.4f}")
                    break
            mlflow.log_metrics({
                'early_stopped_epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_epoch_by_loss': best_epoch_by_loss,
                'best_val_f1': best_val_f1,
                'best_val_iou_at_best_f1': best_val_iou_at_best_f1,
                'best_val_dice_at_best_f1': best_val_dice_at_best_f1,
                'best_val_precision_at_best_f1': best_val_precision_at_best_f1,
                'best_val_recall_at_best_f1': best_val_recall_at_best_f1,
                'best_epoch_by_f1': best_epoch_by_f1,
            })
            best_ckpt = torch.load(best_loss_path, map_location=device)
            model.load_state_dict(best_ckpt['model_state_dict'])
            final_eval = evaluate(model, val_loader, criterion, device, threshold=args.threshold, max_vis=args.num_vis_samples)

        curves_path = artifact_dir / 'training_curves.png'
        history_csv_path = artifact_dir / 'history.csv'
        confusion_path = artifact_dir / 'confusion_matrix.png'
        pred_dir = artifact_dir / 'predictions'
        summary_path = artifact_dir / 'run_summary.txt'
        save_training_curves(history, curves_path)
        save_history_csv(history, history_csv_path)
        save_confusion_matrix(final_eval['confusion'], confusion_path)
        save_prediction_visuals(final_eval['vis_samples'], pred_dir)
        summary = {
            'run_name': args.run_name,
            'mode': args.mode,
            'best_checkpoint_rule': 'lowest_val_loss',
            'report_primary_metric': 'val_f1',
            'best_epoch_by_loss': best_epoch_by_loss,
            'best_val_loss': round(best_val_loss, 6),
            'best_epoch_by_f1': best_epoch_by_f1,
            'best_val_f1': round(best_val_f1, 6),
            'best_val_iou_at_best_f1': round(best_val_iou_at_best_f1, 6),
            'best_val_dice_at_best_f1': round(best_val_dice_at_best_f1, 6),
            'best_val_precision_at_best_f1': round(best_val_precision_at_best_f1, 6),
            'best_val_recall_at_best_f1': round(best_val_recall_at_best_f1, 6),
            'final_eval_loaded_from_best_loss_ckpt_f1': round(final_eval['f1'], 6),
            'final_eval_loaded_from_best_loss_ckpt_iou': round(final_eval['iou'], 6),
            'final_eval_loaded_from_best_loss_ckpt_dice': round(final_eval['dice'], 6),
        }
        save_run_summary(summary, summary_path)
        mlflow.log_artifact(str(best_loss_path))
        mlflow.log_artifact(str(last_path))
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(history_csv_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifacts(str(pred_dir), artifact_path='predictions')
        mlflow.pytorch.log_model(model, artifact_path='model')
        print('\nTraining finished.')
        print(f'Best epoch by loss: {best_epoch_by_loss}')
        print(f'Best val loss: {best_val_loss:.4f}')
        print(f'Best epoch by f1: {best_epoch_by_f1}')
        print(f'Best val f1: {best_val_f1:.4f}')
        print(f'Best val iou at best f1: {best_val_iou_at_best_f1:.4f}')