from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio


class LocalChangeDataset(Dataset):
    """
    Dataset za lokalne bitemporalne Sentinel-2 patcheve
    vraća (t1, t2, mask) tuple:
        t1   : FloatTensor [C, 256, 256]
        t2   : FloatTensor [C, 256, 256]
        mask : FloatTensor [1, 256, 256]  binarna {0.0, 1.0}

    Args:
        patches_dir          : putanja do root foldera sa t1/t2/labels/
        selected_bands       : lista 0-indeksiranih bandova za odabir (None = svi)
        normalize            : scale_10000 za Sentinel,
                               "auto" 
                               "none" 
        apply_imagenet_norm  : True-primjenjuje ImageNet mean/std normalizaciju
                               nakon skaliranja — obavezno za transfer learning od
                               OSCD checkpointa (default: True)
        augment              : "none", "flip", "flip_rot90", "strong"
        patch_size           : veličina izlaznog patcha (default 256)
        split                : "train" (aktivira augmentacije) ili "val"/"test"
    """

    _VALID_AUGMENTS = {"none", "flip", "flip_rot90", "strong"}

    EXTENSIONS = [".tif", ".tiff", ".TIF", ".TIFF"]

    def __init__(
        self,
        patches_dir,
        selected_bands=None,
        normalize="scale_10000",
        apply_imagenet_norm=True,
        augment="none",
        patch_size=256,
        split="val",
    ):
        if augment not in self._VALID_AUGMENTS:
            raise ValueError(
                f"Unsupported augment='{augment}'. "
                f"Use one of: {sorted(self._VALID_AUGMENTS)}."
            )

        self.t1_dir    = Path(patches_dir) / "t1"
        self.t2_dir    = Path(patches_dir) / "t2"
        self.label_dir = Path(patches_dir) / "labels"

        self.selected_bands      = selected_bands
        self.normalize           = normalize
        self.apply_imagenet_norm = apply_imagenet_norm
        self.augment             = augment
        self.patch_size          = patch_size
        self.split               = split

        # ImageNet mean/std koristi se samo ako apply_imagenet_norm=True i C==3
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


        self.ids = self._discover_ids()
        if len(self.ids) == 0:
            raise RuntimeError(
                f"Nema validnih patch parova u {patches_dir}."
            )

    # ---------------------------------------------------------------
    # Discover & find
    # ---------------------------------------------------------------

    def _discover_ids(self):
        import re

        def base(stem):
            return re.sub(r'_(label|t1|t2)$', '', stem)

        label_bases = {}
        for p in sorted(self.label_dir.iterdir()):
            if p.suffix.lower() in [e.lower() for e in self.EXTENSIONS]:
                label_bases[base(p.stem)] = p

        valid = []
        for bid, label_path in label_bases.items():
            t1_found = any(
                (self.t1_dir / (bid + suffix + ext)).exists()
                for suffix in ["_t1", ""]
                for ext in self.EXTENSIONS
            )
            t2_found = any(
                (self.t2_dir / (bid + suffix + ext)).exists()
                for suffix in ["_t2", ""]
                for ext in self.EXTENSIONS
            )
            if t1_found and t2_found:
                valid.append(bid)

        return valid

    def _find_file(self, directory, stem):
        dir_name = directory.name
        suffix_map = {
            "t1":     ["_t1", ""],
            "t2":     ["_t2", ""],
            "labels": ["_label", ""],
        }
        suffixes = suffix_map.get(dir_name, [""])

        for suffix in suffixes:
            for ext in self.EXTENSIONS:
                p = directory / f"{stem}{suffix}{ext}"
                if p.exists():
                    return p

        raise FileNotFoundError(
            f"Nije pronađen fajl za stem='{stem}' u {directory}"
        )

    # ---------------------------------------------------------------
    # Učitavanje
    # ---------------------------------------------------------------

    def _load_image(self, path):
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # [C, H, W]
        if arr.ndim == 2:
            arr = arr[None]
        if self.selected_bands is not None:
            arr = arr[self.selected_bands]
        return arr

    def _load_mask(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        arr = (arr > 0).astype(np.float32)
        return arr[None]  # [1, H, W]

    # ---------------------------------------------------------------
    # Normalizacija
    # ---------------------------------------------------------------

    def _normalize(self, arr):
        if self.normalize == "auto":
            mode = "scale_10000" if arr.max() > 255 else (
                "scale_255" if arr.max() > 1.5 else "none"
            )
        else:
            mode = self.normalize

        if mode == "scale_10000":
            arr = np.clip(arr / 10000.0, 0.0, 1.0)
        elif mode == "scale_255":
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Nepoznat normalize='{mode}'")
        return arr


    # ---------------------------------------------------------------
    # Augmentacije
    # ---------------------------------------------------------------

    def _apply_sync_photometric(self, t1, t2):
        brightness = 1.0 + random.uniform(-0.15, 0.15)
        contrast   = 1.0 + random.uniform(-0.15, 0.15)
        gamma      = 1.0 + random.uniform(-0.15, 0.15)

        def _apply(x):
            x    = x * brightness
            mean = x.mean(dim=(1, 2), keepdim=True)
            x    = (x - mean) * contrast + mean
            x    = torch.clamp(x, 1e-6, 1.0)
            x    = torch.pow(x, gamma)
            return torch.clamp(x, 0.0, 1.0)

        return _apply(t1), _apply(t2)

    def _apply_augment(self, t1, t2, mask):
        if self.augment == "none":
            return t1, t2, mask

        if self.augment in {"flip", "flip_rot90", "strong"}:
            if random.random() < 0.5:
                t1   = torch.flip(t1,   dims=[2])
                t2   = torch.flip(t2,   dims=[2])
                mask = torch.flip(mask, dims=[2])
            if random.random() < 0.5:
                t1   = torch.flip(t1,   dims=[1])
                t2   = torch.flip(t2,   dims=[1])
                mask = torch.flip(mask, dims=[1])

        if self.augment in {"flip_rot90", "strong"}:
            k = random.randint(0, 3)
            if k > 0:
                t1   = torch.rot90(t1,   k=k, dims=[1, 2])
                t2   = torch.rot90(t2,   k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])

        if self.augment == "strong":
            if random.random() < 0.8:
                t1, t2 = self._apply_sync_photometric(t1, t2)

        return t1, t2, mask

    # ---------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx % len(self.ids)]

        t1_path    = self._find_file(self.t1_dir,    stem)
        t2_path    = self._find_file(self.t2_dir,    stem)
        label_path = self._find_file(self.label_dir, stem)

        t1   = torch.from_numpy(self._normalize(self._load_image(t1_path))).float()
        t2   = torch.from_numpy(self._normalize(self._load_image(t2_path))).float()
        mask = torch.from_numpy(self._load_mask(label_path)).float()

        if self.split == "train":
            t1, t2, mask = self._apply_augment(t1, t2, mask)

        if self.apply_imagenet_norm and t1.shape[0] == 3:
            t1 = (t1 - self.mean) / self.std
            t2 = (t2 - self.mean) / self.std

        return t1, t2, mask

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @property
    def num_channels(self):
        arr = self._load_image(self._find_file(self.t1_dir, self.ids[0]))
        return arr.shape[0]

    def summary(self):
        total = len(self.ids)
        positive, empty = 0, 0
        change_px, total_px = 0, 0
        for stem in self.ids:
            mask = self._load_mask(self._find_file(self.label_dir, stem))
            if mask.sum() > 0:
                positive += 1
            else:
                empty += 1
            change_px += int(mask.sum())
            total_px  += mask.size

        print("=" * 44)
        print(f"LocalChangeDataset — {self.t1_dir.parent}")
        print("=" * 44)
        print(f"  Ukupno patches     : {total}")
        print(f"  Virtualni len()    : {len(self)}")
        print(f"  Pozitivni          : {positive}")
        print(f"  Negativni          : {empty}")
        print(f"  Change pikseli     : {change_px / total_px * 100:.2f}%")
        print(f"  neg:pos            : {(total_px - change_px) / max(change_px, 1):.0f}:1")
        print(f"  Broj kanala        : {self.num_channels}")
        print(f"  Normalizacija      : {self.normalize}")
        print(f"  ImageNet norm      : {self.apply_imagenet_norm}")
        print(f"  Augment            : {self.augment}")
        print("=" * 44)