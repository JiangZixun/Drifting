from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_CLASS_NAMES = [
    "Clr",
    "Ci",
    "Cs",
    "DC",
    "Ac",
    "As",
    "Ns",
    "Cu",
    "Sc",
    "St",
]

DEFAULT_MINMAX_16 = {
    "min": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        207.1699981689453,
        187.04998779296875,
        183.6999969482422,
        183.8599853515625,
        182.22999572753906,
        208.33999633789062,
        183.83999633789062,
        181.66000366210938,
        181.54998779296875,
        184.88999938964844,
    ],
    "max": [
        1.2105,
        1.21420002,
        1.19639993,
        1.22599995,
        1.23109996,
        1.23199999,
        370.76998901,
        261.57998657,
        271.57998657,
        274.79998779,
        313.61999512,
        281.30999756,
        317.63000488,
        317.38998413,
        310.6000061,
        283.77999878,
    ],
}

DEFAULT_MINMAX_17 = {
    "min": [0.0, *DEFAULT_MINMAX_16["min"]],
    "max": [95.37999725341797, *DEFAULT_MINMAX_16["max"]],
}


@dataclass
class NormalizationConfig:
    mode: str = "sample_minmax"
    dataset_min: Optional[List[float]] = None
    dataset_max: Optional[List[float]] = None
    clip: bool = True


class ChannelNormalizer:
    def __init__(self, config: NormalizationConfig):
        self.config = config

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = np.nan_to_num(image.astype(np.float32), copy=False)
        if self.config.mode == "dataset_minmax":
            if self.config.dataset_min is None or self.config.dataset_max is None:
                raise ValueError("dataset_minmax normalization requires dataset_min and dataset_max")
            min_val = np.asarray(self.config.dataset_min, dtype=np.float32)
            max_val = np.asarray(self.config.dataset_max, dtype=np.float32)
            if min_val.shape[0] != image.shape[0] or max_val.shape[0] != image.shape[0]:
                raise ValueError(
                    f"Normalization channel mismatch: image has {image.shape[0]} channels, "
                    f"dataset_min has {min_val.shape[0]}, dataset_max has {max_val.shape[0]}"
                )
            min_val = min_val[:, None, None]
            max_val = max_val[:, None, None]
            image = (image - min_val) / np.maximum(max_val - min_val, 1e-6)
        elif self.config.mode == "sample_minmax":
            min_val = image.min(axis=(1, 2), keepdims=True)
            max_val = image.max(axis=(1, 2), keepdims=True)
            image = (image - min_val) / np.maximum(max_val - min_val, 1e-6)
        elif self.config.mode == "identity":
            pass
        else:
            raise ValueError(f"Unsupported normalization mode: {self.config.mode}")

        if self.config.clip:
            image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class RandomFlip:
    def __init__(self, flip_ratio: float = 0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        sample = random.random()
        if sample < self.flip_ratio / 2:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        elif sample < self.flip_ratio:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask


class RandomRotation:
    def __call__(self, image: np.ndarray, mask: np.ndarray):
        mode = np.random.randint(0, 4)
        if mode == 0:
            return image, mask
        image = np.rot90(image, k=mode, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=mode).copy()
        return image, mask


class PadToSize:
    def __init__(self, size: int, image_pad_value: float = 0.0, mask_pad_value: int = 255):
        self.size = size
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        _, height, width = image.shape
        if height > self.size or width > self.size:
            raise ValueError(f"Input size {(height, width)} exceeds target pad size {self.size}")
        pad_h = self.size - height
        pad_w = self.size - width
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h == 0 and pad_w == 0:
            return image, mask
        image = np.pad(
            image,
            ((0, 0), (top, bottom), (left, right)),
            mode="constant",
            constant_values=self.image_pad_value,
        )
        mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=self.mask_pad_value)
        return image, mask


class MMapCloudSegStore:
    def __init__(self, root: str):
        mmap_root = os.path.join(root, "mmap")
        manifest_path = os.path.join(mmap_root, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Expected memmap manifest under {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.root = mmap_root
        self.length = int(manifest["sample_count"])
        self.image_shape = tuple(manifest["image_shape"])
        self.label_shape = tuple(manifest["label_shape"])
        self.shards = manifest["shards"]
        self.sample_ids = manifest.get("sample_ids", [])
        self.source_paths = manifest.get("source_paths", [])
        self._prefix = []
        offset = 0
        for shard in self.shards:
            self._prefix.append(offset)
            offset += int(shard["count"])
        self._image_arrays: Dict[int, np.memmap] = {}
        self._label_arrays: Dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.length

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        shard_idx = 0
        left, right = 0, len(self._prefix) - 1
        while left <= right:
            mid = (left + right) // 2
            if self._prefix[mid] <= index:
                shard_idx = mid
                left = mid + 1
            else:
                right = mid - 1
        return shard_idx, index - self._prefix[shard_idx]

    def _image_memmap(self, shard_idx: int) -> np.memmap:
        array = self._image_arrays.get(shard_idx)
        if array is None:
            shard = self.shards[shard_idx]
            path = os.path.join(self.root, shard["image_file"])
            array = np.load(path, mmap_mode="r")
            self._image_arrays[shard_idx] = array
        return array

    def _label_memmap(self, shard_idx: int) -> np.memmap:
        array = self._label_arrays.get(shard_idx)
        if array is None:
            shard = self.shards[shard_idx]
            path = os.path.join(self.root, shard["label_file"])
            array = np.load(path, mmap_mode="r")
            self._label_arrays[shard_idx] = array
        return array

    def get(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        shard_idx, local_idx = self._locate(index)
        image = self._image_memmap(shard_idx)[local_idx]
        label = self._label_memmap(shard_idx)[local_idx]
        return image, label

    def get_metadata(self, index: int) -> Dict[str, str]:
        sample_id = self.sample_ids[index] if self.sample_ids else str(index)
        source_path = self.source_paths[index] if self.source_paths else sample_id
        return {
            "sample_id": sample_id,
            "datetime": parse_sample_datetime(sample_id),
            "source_path": source_path,
        }


_SAMPLE_DATETIME_RE = re.compile(r".*_(\d{8})_(\d{4})$")


def parse_sample_datetime(sample_id: str) -> str:
    stem = os.path.splitext(os.path.basename(sample_id))[0]
    match = _SAMPLE_DATETIME_RE.match(stem)
    if match is None:
        return ""
    return f"{match.group(1)}_{match.group(2)}"


class CloudSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        num_classes: int = 10,
        ignore_index: int = 10,
        class_names: Optional[List[str]] = None,
        normalization: Optional[Dict] = None,
        transforms=None,
    ):
        self.files: Sequence[str] = []
        self.mmap_store: Optional[MMapCloudSegStore] = None

        mmap_manifest = os.path.join(root, "mmap", "manifest.json")
        data_dir = os.path.join(root, "data")
        if os.path.isfile(mmap_manifest):
            self.mmap_store = MMapCloudSegStore(root)
        elif os.path.isdir(data_dir):
            self.files = sorted(os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".npz"))
            if not self.files:
                raise RuntimeError(f"No .npz files found under {data_dir}")
        else:
            raise FileNotFoundError(
                f"Expected either memmap dataset under {mmap_manifest} or npz files under {data_dir}"
            )

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or list(DEFAULT_CLASS_NAMES)
        self.transforms = transforms
        norm_cfg = normalization or {
            "mode": "dataset_minmax",
            "dataset_min": list(DEFAULT_MINMAX_17["min"]),
            "dataset_max": list(DEFAULT_MINMAX_17["max"]),
        }
        self.image_normalizer = ChannelNormalizer(NormalizationConfig(**norm_cfg))

    def __len__(self) -> int:
        if self.mmap_store is not None:
            return len(self.mmap_store)
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.mmap_store is not None:
            image, mask = self.mmap_store.get(index)
            metadata = self.mmap_store.get_metadata(index)
            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.int64)
        else:
            file_path = self.files[index]
            sample = np.load(file_path)
            image = sample["image"].astype(np.float32)
            mask = sample["label"].astype(np.int64)
            sample_id = os.path.splitext(os.path.basename(file_path))[0]
            metadata = {
                "sample_id": sample_id,
                "datetime": parse_sample_datetime(sample_id),
                "source_path": file_path,
            }

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        image = self.image_normalizer(image)
        image = torch.clamp(image, min=0.0)
        mask = torch.from_numpy(mask.copy()).long()
        invalid = (mask < 0) | (mask >= self.num_classes)
        mask[invalid] = self.ignore_index
        return {
            "pixel_values": image,
            "labels": mask,
            "sample_id": metadata["sample_id"],
            "datetime": metadata["datetime"],
            "source_path": metadata["source_path"],
        }


def make_transforms(train: bool, pad_to_size: Optional[int], ignore_index: int):
    transforms = []
    if train:
        transforms.extend([RandomRotation(), RandomFlip()])
    if pad_to_size is not None:
        transforms.append(PadToSize(pad_to_size, mask_pad_value=ignore_index))
    if not transforms:
        return None
    return Compose(transforms)


def build_dataloaders(
    data_root: str,
    train_split: str = "train1",
    val_split: str = "val1",
    *,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    num_workers: int = 4,
    ignore_index: int = 10,
    normalization: Optional[Dict] = None,
    pad_to_size: Optional[int] = None,
):
    train_ds = CloudSegmentationDataset(
        os.path.join(data_root, train_split),
        ignore_index=ignore_index,
        normalization=normalization,
        transforms=make_transforms(train=True, pad_to_size=pad_to_size, ignore_index=ignore_index),
    )
    val_ds = CloudSegmentationDataset(
        os.path.join(data_root, val_split),
        ignore_index=ignore_index,
        normalization=normalization,
        transforms=make_transforms(train=False, pad_to_size=pad_to_size, ignore_index=ignore_index),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds
