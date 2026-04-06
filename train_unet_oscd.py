import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from src.oscd import OSCDDataset

def set_seed(seed:int):
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
            nn.ReLU(inplace=True)
        )    

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(ch, feature))
            ch = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] *2)
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
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
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
    

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_pixel_acc = 0.0
    batches = 0

    all_preds = []
    all_targets = []
    vis_samples = []

    for t1, t2, mask in loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        mask = mask.to(device)
        x = torch.cat([t1,t2], dim=1)

        logits = model(x)
        loss = criterion(logits, mask)
        probs = torch.sigmoid(logits)
        preds = (probs>0.5).float()

        tp = (preds*mask).sum(dim=(1,2,3))
        fp = (preds*(1-mask)).sum(dim=(1,2,3))
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

        if len(vis_samples) < 3:
            for i in range(min(t1.size(0), 3 - len(vis_samples))):
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

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    return {
        "loss": total_loss / max(batches, 1),
        "iou": total_iou / max(batches, 1),
        "dice": total_dice / max(batches, 1),
        "precision": total_precision / max(batches, 1),
        "recall": total_recall / max(batches, 1),
        "f1": total_f1 / max(batches, 1),
        "pixel_accuracy": total_pixel_acc / max(batches, 1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "vis_samples": vis_samples,
    }

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


def main(args):
    set_seed(args.seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    train_ds = OSCDDataset(
        split="train",
        patch_size=args.patch_size,
        crop_mode=args.train_mode,
        augment=args.augment,
    )
    val_ds = OSCDDataset(
        split="test",
        patch_size=args.patch_size,
        crop_mode=args.val_mode,
        augment="none",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(in_channels=6, out_channels=1, features=(32, 64, 128, 256)).to(device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_path = out_dir / "best_unet_oscd.pt"
    last_path = out_dir / "last_unet_oscd.pt"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
    }

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
                "bce_weight": args.bce_weight,
                "seed": args.seed,
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
            }
        )

        early_stopping = EarlyStopping(patience=10, min_delta=0.001)

        epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])
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

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_iou = val_metrics["iou"]
                best_val_dice = val_metrics["dice"]
                best_epoch = epoch
                torch.save(checkpoint, best_path)
                print(f"Saved new best model to {best_path}")

            if early_stopping.step(val_metrics["loss"], epoch):
                print(
                    f"Early stopping at epoch {epoch:03d}. "
                    f"Best epoch was {early_stopping.best_epoch:03d} "
                    f"with val_loss={early_stopping.best_loss:.4f}"
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
        final_eval = evaluate(model, val_loader, criterion, device)

        curves_path = artifact_dir / "training_curves.png"
        save_training_curves(history, curves_path)

        confusion_path = artifact_dir / "confusion_matrix.png"
        save_confusion_matrix(final_eval["confusion"], confusion_path)

        pred_dir = artifact_dir / "predictions"
        save_prediction_visuals(final_eval["vis_samples"], pred_dir)

        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(last_path))
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifacts(str(pred_dir), artifact_path="predictions")
        mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train U-Net on OSCD RGB with MLflow logging (CPU only)"
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--train-mode", type=str, default="random_crop")
    parser.add_argument("--val-mode", type=str, default="center_crop")
    parser.add_argument("--augment", type=str, default="none")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--experiment-name", type=str, default="oscd_change_detection")
    parser.add_argument("--run-name", type=str, default="baseline_unet_cpu")
    parser.add_argument("--model-name", type=str, default="unet")
    parser.add_argument("--dataset-name", type=str, default="blanchon/OSCD_RGB")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="file:./mlruns")
    args = parser.parse_args()
    main(args)