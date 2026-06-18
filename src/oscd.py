from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import random


class OSCDDataset(Dataset):
    # Supported augment modes:
    #   none– no augmentation
    #   flip- random horizontal + vertical flip
    #   flip_rot90– flip + random rot90 (k in {0,1,2,3})
    #   strong– flip_rot90 + synchronised brightness/contrast/gamma
    _VALID_AUGMENTS = {"none", "flip", "flip_rot90", "strong"}

    def __init__(
        self,
        split="train",
        hf_name="blanchon/OSCD_RGB",
        patch_size=256,
        crop_mode="random_crop",
        augment="none",
    ):
        if augment not in self._VALID_AUGMENTS:
            raise ValueError(
                f"Unsupported augment='{augment}'. "
                f"Use one of: {sorted(self._VALID_AUGMENTS)}."
            )
        self.split = split
        self.ds = load_dataset(hf_name)[split]
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.augment = augment
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.ds)

    def _to_tensor_image(self, image):
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr).float()

        return x

    def _to_tensor_mask(self, mask):
        arr = np.array(mask, dtype=np.float32)

        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] != 1:
            arr = arr[..., 0][None, ...]

        arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(arr).float()

    def _crop_with_pad(self, x, top, left):
        _, h, w = x.shape
        ps = self.patch_size

        pad_h = max(0, top + ps - h)
        pad_w = max(0, left + ps - w)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return x[:, top:top + ps, left:left + ps]

    def _get_change_aware_crop(self, mask, h, w):
        ps = self.patch_size

        mask_np = np.array(mask).astype(np.float32)
        change_coords = np.argwhere(mask_np > 0)

        if len(change_coords) > 0 and random.random() < 0.3:
            idx = random.randint(0, len(change_coords) - 1)
            y, x = change_coords[idx]

            top = max(0, min(y - ps // 2, h - ps))
            left = max(0, min(x - ps // 2, w - ps))
        else:
            top = 0 if h <= ps else random.randint(0, h - ps)
            left = 0 if w <= ps else random.randint(0, w - ps)

        return top, left

    def _get_crop_coords(self, h, w):
        ps = self.patch_size

        if self.crop_mode == "center_crop":
            top = max((h - ps) // 2, 0)
            left = max((w - ps) // 2, 0)
        elif self.crop_mode == "random_crop":
            top = 0 if h <= ps else random.randint(0, h - ps)
            left = 0 if w <= ps else random.randint(0, w - ps)
        else:
            raise ValueError("Unsupported crop_mode")

        return top, left

    def _apply_sync_photometric(self, t1, t2):
        brightness = 1.0 + random.uniform(-0.15, 0.15)
        contrast = 1.0 + random.uniform(-0.15, 0.15)
        gamma = 1.0 + random.uniform(-0.15, 0.15)

        def _apply(x):
            x = x * brightness
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = (x - mean) * contrast + mean
            x = torch.clamp(x, 1e-6, 1.0)
            x = torch.pow(x, gamma)
            return torch.clamp(x, 0.0, 1.0)

        return _apply(t1), _apply(t2)

    def _apply_augment(self, t1, t2, mask):
        if self.augment == "none":
            return t1, t2, mask

        if self.augment in {"flip", "flip_rot90", "strong"}:
            if random.random() < 0.5:
                t1 = torch.flip(t1, dims=[2])
                t2 = torch.flip(t2, dims=[2])
                mask = torch.flip(mask, dims=[2])

            if random.random() < 0.5:
                t1 = torch.flip(t1, dims=[1])
                t2 = torch.flip(t2, dims=[1])
                mask = torch.flip(mask, dims=[1])

        if self.augment in {"flip_rot90", "strong"}:
            k = random.randint(0, 3)
            if k > 0:
                t1 = torch.rot90(t1, k=k, dims=[1, 2])
                t2 = torch.rot90(t2, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])

        if self.augment == "strong":
            if random.random() < 0.8:
                t1, t2 = self._apply_sync_photometric(t1, t2)

        return t1, t2, mask

    def __getitem__(self, idx):
        item = self.ds[idx]

        t1 = self._to_tensor_image(item["image1"])
        t2 = self._to_tensor_image(item["image2"])
        mask = self._to_tensor_mask(item["mask"])

        _, h, w = t1.shape

        top, left = self._get_change_aware_crop(mask, h, w)

        t1 = self._crop_with_pad(t1, top, left)
        t2 = self._crop_with_pad(t2, top, left)
        mask = self._crop_with_pad(mask, top, left)

        if self.split == "train":
            t1, t2, mask = self._apply_augment(t1, t2, mask)

        t1 = (t1 - self.mean) / self.std
        t2 = (t2 - self.mean) / self.std

        return t1, t2, mask