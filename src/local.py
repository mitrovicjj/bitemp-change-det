from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from PIL import Image

class LocalChangeDataset(Dataset):
    """
    Dataset za lokalne bitemporalne Sentinel-2 patch-eve.

    Očekivana struktura direktorijuma:
        patches_dir/
        ├── t1/        <- slike T1 (GeoTIF ili PNG, RGB ili MSI)
        ├── t2/        <- slike T2 (isti format i ime kao T1)
        └── labels/    <- binarne maske promjene (GeoTIF ili PNG)

    Imenovanje: fajlovi u t1/, t2/ i labels/ moraju imati isti stem
    (npr. patch_0001.tif mora postojati u sva 3 direktorijuma).

    Interface je identičan OSCDDataset — vraća (t1, t2, mask) tuple:
        t1   : FloatTensor [C, 256, 256]  normalizovano [0, 1]
        t2   : FloatTensor [C, 256, 256]  normalizovano [0, 1]
        mask : FloatTensor [1, 256, 256]  binarna {0.0, 1.0}

    Args:
        patches_dir   : putanja do root direktorijuma sa t1/t2/labels/
        selected_bands: lista 0-indeksiranih bandova za odabir (None = svi)
        normalize     : "scale_10000" za Sentinel-2 L1C/L2A (dijeli sa 10000),
                        "scale_255"   za uint8 PNG (dijeli sa 255),
                        "per_image"   za per-kanal min-max normalizaciju,
                        "auto"        automatski detektuje na osnovu vrijednosti
        augment       : "none", "flip", "flip_rot90"
        patch_size    : veličina izlaznog patcha (default 256)
        crop_mode     : "none" (patch se koristi as-is), "center_crop", "random_crop"
        split         : "train" (aktivira augmentacije) ili "val"/"test"
    """

    EXTENSIONS = [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG"]

    def __init__(
        self,
        patches_dir,
        selected_bands=None,
        normalize="scale_10000",
        augment="none",
        patch_size=256,
        crop_mode="none",
        split="val",
    ):
        if rasterio is None and Image is None:
            raise ImportError("pip install rasterio pillow")

        self.t1_dir = Path(patches_dir) / "t1"
        self.t2_dir = Path(patches_dir) / "t2"
        self.label_dir = Path(patches_dir) / "labels"

        for d in [self.t1_dir, self.t2_dir, self.label_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Direktorijum ne postoji: {d}")

        self.selected_bands = selected_bands
        self.normalize = normalize
        self.augment = augment
        self.patch_size = patch_size
        self.crop_mode = crop_mode
        self.split = split

        self.ids = self._discover_ids()
        if len(self.ids) == 0:
            raise RuntimeError(
                f"Nema validnih patch parova u {patches_dir}."
            )
    def _discover_ids(self):
        import re

        def base(stem):
            # ukloni _label, _t1, _t2 sa kraja stema
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
            "t1": ["_t1", ""],
            "t2": ["_t2", ""],
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
        path = Path(path)
        if path.suffix.lower() in (".tif", ".tiff"):
            return self._load_tif(path)
        raise ValueError(f"Nepodržan format slike: {path.suffix}. Očekujem samo .tif ili .tiff.")

    def _load_tif(self, path):
        if rasterio is None:
            raise ImportError("rasterio nije instaliran: pip install rasterio")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # [C, H, W]
        if arr.ndim == 2:
            arr = arr[None]
        if self.selected_bands is not None:
            arr = arr[self.selected_bands]
        return arr

    def _load_mask(self, path):
        path = Path(path)
        if path.suffix.lower() in (".tif", ".tiff"):
            if rasterio is None:
                raise ImportError("rasterio nije instaliran: pip install rasterio")
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
        else:
            raise ValueError(f"Nepodržan format maske: {path.suffix}. Očekujem samo .tif ili .tiff.")

        if arr.ndim == 3:
            arr = arr[..., 0]

        arr = (arr > 0).astype(np.float32)
        return arr[None]

    # ---------------------------------------------------------------
    # Normalizacija
    # ---------------------------------------------------------------

    def _normalize(self, arr):
        if self.normalize == "auto":
            if arr.max() > 1.5:
                mode = "scale_10000" if arr.max() > 255 else "scale_255"
            else:
                mode = "none"
        else:
            mode = self.normalize

        if mode == "scale_10000":
            arr = np.clip(arr / 10000.0, 0.0, 1.0)
        elif mode == "scale_255":
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        elif mode == "per_image":
            out = np.empty_like(arr)
            for c in range(arr.shape[0]):
                band = arr[c]
                mn, mx = band.min(), band.max()
                out[c] = (band - mn) / (mx - mn + 1e-8)
            arr = out
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Nepoznat normalize='{mode}'")
        return arr

    # ---------------------------------------------------------------
    # Crop i augmentacije — identično kao OSCDDataset
    # ---------------------------------------------------------------

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
            return max((h - ps) // 2, 0), max((w - ps) // 2, 0)
        elif self.crop_mode == "random_crop":
            top = 0 if h <= ps else random.randint(0, h - ps)
            left = 0 if w <= ps else random.randint(0, w - ps)
            return top, left
        elif self.crop_mode == "none":
            return 0, 0
        raise ValueError(f"Nepoznat crop_mode='{self.crop_mode}'")

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

    # ---------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        t1_path = self._find_file(self.t1_dir, stem)
        t2_path = self._find_file(self.t2_dir, stem)
        label_path = self._find_file(self.label_dir, stem)

        t1 = torch.from_numpy(self._normalize(self._load_image(t1_path))).float()
        t2 = torch.from_numpy(self._normalize(self._load_image(t2_path))).float()
        mask = torch.from_numpy(self._load_mask(label_path)).float()

        _, h, w = t1.shape
        if self.crop_mode != "none" or h != self.patch_size or w != self.patch_size:
            top, left = self._get_crop_coords(h, w)
            t1 = self._crop_with_pad(t1, top, left)
            t2 = self._crop_with_pad(t2, top, left)
            mask = self._crop_with_pad(mask, top, left)

        if self.split == "train":
            t1, t2, mask = self._apply_augment(t1, t2, mask)

        return t1, t2, mask

    # ---------------------------------------------------------------
    # Helper metode
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
            total_px += mask.size

        print("=" * 40)
        print(f"LocalChangeDataset — {self.t1_dir.parent}")
        print("=" * 40)
        print(f"  Ukupno patches:     {total}")
        print(f"  Pozitivni: {positive}")
        print(f"  Negativni:   {empty}")
        print(f"  Change pikseli:       {change_px / total_px * 100:.2f}%")
        print(f"  neg:pos:  {(total_px - change_px) / max(change_px, 1):.0f}:1")
        print(f"  Broj kanala:          {self.num_channels}")
        print(f"  Normalizacija:        {self.normalize}")
        print("=" * 40)