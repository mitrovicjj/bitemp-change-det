from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import random
import io

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except ImportError:
    pa = None
    ipc = None


class OSCDDataset(Dataset):
    def __init__(
        self,
        split="train",
        hf_name="blanchon/OSCD_MSI",
        patch_size=256,
        crop_mode="random_crop",
        augment="none",
        normalize="scale_10000",
        selected_bands=None,
    ):
        self.split = split
        self.ds = load_dataset(hf_name)[split]
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.augment = augment
        self.normalize = normalize
        self.selected_bands = selected_bands

        if pa is None or ipc is None:
            raise ImportError("pyarrow is required for Arrow-backed OSCD_MSI loading.")

        self.arrow_path = self.ds.cache_files[0]["filename"]
        with pa.memory_map(self.arrow_path, "r") as source:
            reader = ipc.open_stream(source)
            self.table = reader.read_all()

    def __len__(self):
        return self.table.num_rows

    @property
    def num_bands(self):
        if self.selected_bands is not None:
            return len(self.selected_bands)
        return 13

    @property
    def input_channels(self):
        return 2 * self.num_bands

    def _decode_tiff_bytes(self, obj):
        if isinstance(obj, dict) and "bytes" in obj:
            b = obj["bytes"]
            if tifffile is None:
                raise ImportError("Install tifffile to decode TIFF bytes: pip install tifffile")
            return np.asarray(tifffile.imread(io.BytesIO(b)))
        return np.asarray(obj)

    def _get_arrow_cell(self, col_name, idx):
        col = self.table.column(col_name)
        return col[ idx ].as_py()

    def _load_image_from_arrow(self, col_name, idx):
        value = self._get_arrow_cell(col_name, idx)

        if isinstance(value, dict) and "bytes" in value:
            arr = self._decode_tiff_bytes(value)
        else:
            arr = np.asarray(value)

        return arr

    def _load_mask_from_arrow(self, idx):
        value = self._get_arrow_cell("mask", idx)

        if isinstance(value, list):
            value = dict(value)

        arr = self._decode_tiff_bytes(value)
        return arr

    def _to_tensor_image(self, arr):
        arr = np.asarray(arr)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape={arr.shape}")

        if arr.shape[0] in (1, 3, 13):
            pass
        elif arr.shape[-1] in (1, 3, 13):
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Cannot infer channel axis from shape={arr.shape}")

        if self.selected_bands is not None:
            arr = arr[self.selected_bands]

        arr = arr.astype(np.float32)

        if self.normalize == "scale_10000":
            arr = arr / 10000.0
            arr = np.clip(arr, 0.0, 1.5)
        elif self.normalize == "per_image":
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(arr.shape[0]):
                band = arr[c]
                mn = band.min()
                mx = band.max()
                out[c] = (band - mn) / (mx - mn) if mx > mn else band
            arr = out
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalize='{self.normalize}'")

        return torch.from_numpy(arr).float()

    def _to_tensor_mask(self, arr):
        arr = np.asarray(arr)

        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] == 1:
            pass
        elif arr.ndim == 3:
            arr = arr[..., 0][None, ...]
        else:
            raise ValueError(f"Unexpected mask shape={arr.shape}")

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
                f"Unsupported crop_mode='{self.crop_mode}'. Use 'random_crop' or 'center_crop'."
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
        img1 = self._load_image_from_arrow("image1", idx)
        img2 = self._load_image_from_arrow("image2", idx)
        msk = self._load_mask_from_arrow(idx)

        t1 = self._to_tensor_image(img1)
        t2 = self._to_tensor_image(img2)
        mask = self._to_tensor_mask(msk)

        _, h, w = t1.shape
        top, left = self._get_crop_coords(h, w)

        t1 = self._crop_with_pad(t1, top, left)
        t2 = self._crop_with_pad(t2, top, left)
        mask = self._crop_with_pad(mask, top, left)

        if self.split == "train":
            t1, t2, mask = self._apply_augment(t1, t2, mask)

        return t1, t2, mask