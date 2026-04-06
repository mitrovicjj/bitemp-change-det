from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import random


class OSCDDataset(Dataset):
    def __init__(
        self,
        split="train",
        hf_name="blanchon/OSCD_RGB",
        patch_size=256,
        crop_mode="random_crop",
        augment="none",
    ):
        self.split = split
        self.ds = load_dataset(hf_name)[split]
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.augment = augment

    def __len__(self):
        return len(self.ds)

    def _to_tensor_image(self, image):
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()

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

    def _get_crop_coords(self, h, w):
        ps = self.patch_size

        if self.crop_mode == "center_crop":
            top = max((h - ps) // 2, 0)
            left = max((w - ps) // 2, 0)
        elif self.crop_mode == "random_crop":
            top = 0 if h <= ps else random.randint(0, h - ps)
            left = 0 if w <= ps else random.randint(0, w - ps)
        else:
            raise ValueError(
                f"Unsupported crop_mode='{self.crop_mode}'. "
                f"Use 'random_crop' or 'center_crop'."
            )

        return top, left

    def _apply_augment(self, t1, t2, mask):
        if self.augment == "none":
            return t1, t2, mask

        if self.augment in {"flip", "flip_rot90"}:
            if random.random() < 0.5:
                t1 = torch.flip(t1, dims=[2])
                t2 = torch.flip(t2, dims=[2])
                mask = torch.flip(mask, dims=[2])

            if random.random() < 0.5:
                t1 = torch.flip(t1, dims=[1])
                t2 = torch.flip(t2, dims=[1])
                mask = torch.flip(mask, dims=[1])

        if self.augment == "flip_rot90":
            k = random.randint(0, 3)
            if k > 0:
                t1 = torch.rot90(t1, k=k, dims=[1, 2])
                t2 = torch.rot90(t2, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])

        return t1, t2, mask

    def __getitem__(self, idx):
        item = self.ds[idx]

        t1 = self._to_tensor_image(item["image1"])
        t2 = self._to_tensor_image(item["image2"])
        mask = self._to_tensor_mask(item["mask"])

        _, h, w = t1.shape
        top, left = self._get_crop_coords(h, w)

        t1 = self._crop_with_pad(t1, top, left)
        t2 = self._crop_with_pad(t2, top, left)
        mask = self._crop_with_pad(mask, top, left)

        if self.split == "train":
            t1, t2, mask = self._apply_augment(t1, t2, mask)

        return t1, t2, mask