from __future__ import annotations

import bisect
import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for .h5 big dataset
    h5py = None


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


def _resolve_gradient_normalization(config: Optional[Dict]) -> Dict:
    if config is None:
        return {"mode": "sample_minmax", "clip": True}
    return dict(config)


def _conv2d_same(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    padded = np.pad(channel, ((1, 1), (1, 1)), mode="edge")
    output = np.zeros((height, width), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            output += kernel[row, col] * padded[row : row + height, col : col + width]
    return output


def _sobel_gradients(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kernel_x = np.asarray(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    kernel_y = np.asarray(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    grad_x = _conv2d_same(channel, kernel_x)
    grad_y = _conv2d_same(channel, kernel_y)
    return grad_x, grad_y


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
        if not os.path.isdir(mmap_root):
            raise FileNotFoundError(f"Expected mmap directory under {mmap_root}")
        manifest_path = os.path.join(mmap_root, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = self._build_manifest_from_shards(mmap_root)

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

    @staticmethod
    def _build_manifest_from_shards(mmap_root: str) -> Dict[str, Any]:
        image_files = sorted(name for name in os.listdir(mmap_root) if re.fullmatch(r"images-\d+\.npy", name))
        label_files = sorted(name for name in os.listdir(mmap_root) if re.fullmatch(r"labels-\d+\.npy", name))
        if not image_files or not label_files:
            raise FileNotFoundError(
                f"Expected mmap manifest or shard files under {mmap_root}, "
                "but found neither manifest.json nor images-*.npy/labels-*.npy"
            )

        labels_by_suffix = {
            name[len("labels-") : -len(".npy")]: name
            for name in label_files
        }
        shard_names = []
        for image_name in image_files:
            suffix = image_name[len("images-") : -len(".npy")]
            label_name = labels_by_suffix.get(suffix)
            if label_name is not None:
                shard_names.append((suffix, image_name, label_name))
        if not shard_names:
            raise RuntimeError(
                f"No matched mmap shard pairs found under {mmap_root}. "
                "Expected images-XXXXX.npy and labels-XXXXX.npy with same suffix."
            )

        shards: List[Dict[str, Any]] = []
        sample_count = 0
        image_shape = None
        label_shape = None
        image_dtype = None
        label_dtype = None

        for idx, (_, image_name, label_name) in enumerate(shard_names):
            image_path = os.path.join(mmap_root, image_name)
            label_path = os.path.join(mmap_root, label_name)
            try:
                image_array = np.load(image_path, mmap_mode="r")
                label_array = np.load(label_path, mmap_mode="r")
            except (ValueError, OSError) as exc:
                warnings.warn(
                    f"Skipping incomplete/corrupted mmap shard pair {image_name}/{label_name}: {exc}",
                    RuntimeWarning,
                )
                continue
            if image_array.ndim != 4 or label_array.ndim != 3:
                warnings.warn(
                    f"Skipping shard pair with unexpected shapes {image_name}/{label_name}: "
                    f"image {image_array.shape}, label {label_array.shape}. "
                    "Expected [N,C,H,W] and [N,H,W].",
                    RuntimeWarning,
                )
                continue
            if image_array.shape[0] != label_array.shape[0]:
                warnings.warn(
                    f"Skipping shard pair with mismatched sample count {image_name}/{label_name}: "
                    f"{image_array.shape[0]} vs {label_array.shape[0]}",
                    RuntimeWarning,
                )
                continue
            if image_shape is None:
                image_shape = tuple(image_array.shape[1:])
                label_shape = tuple(label_array.shape[1:])
                image_dtype = str(image_array.dtype)
                label_dtype = str(label_array.dtype)
            else:
                if tuple(image_array.shape[1:]) != image_shape or tuple(label_array.shape[1:]) != label_shape:
                    warnings.warn(
                        f"Skipping shard pair with inconsistent shape {image_name}/{label_name}: "
                        f"image {image_array.shape[1:]} label {label_array.shape[1:]}, "
                        f"expected {image_shape}/{label_shape}",
                        RuntimeWarning,
                    )
                    continue
            count = int(image_array.shape[0])
            shards.append(
                {
                    "index": idx,
                    "count": count,
                    "image_file": image_name,
                    "label_file": label_name,
                }
            )
            sample_count += count
            del image_array, label_array

        if not shards:
            raise RuntimeError(
                f"No readable mmap shard pairs found under {mmap_root}. "
                "Transfer may still be incomplete."
            )

        return {
            "sample_count": sample_count,
            "image_shape": list(image_shape),
            "label_shape": list(label_shape),
            "image_dtype": image_dtype,
            "label_dtype": label_dtype,
            "sample_ids": [],
            "source_paths": [],
            "shards": shards,
        }

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
        if self.sample_ids:
            sample_id = self.sample_ids[index]
            source_path = self.source_paths[index] if self.source_paths else sample_id
        else:
            shard_idx, local_idx = self._locate(index)
            sample_id = f"mmap_s{shard_idx:05d}_{local_idx:06d}"
            source_path = os.path.join(self.root, self.shards[shard_idx]["image_file"])
        return {
            "sample_id": sample_id,
            "datetime": parse_sample_datetime(sample_id),
            "source_path": source_path,
        }


_SAMPLE_DATETIME_RE = re.compile(r".*_(\d{8})_(\d{4})(?:_.*)?$")


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
        input_channel_indices: Optional[List[int]] = None,
        feature_augmentation: Optional[Dict] = None,
        h5_patch_size: int = 256,
        transforms=None,
    ):
        self.files: Sequence[str] = []
        self.h5_files: Sequence[str] = []
        self.mmap_store: Optional[MMapCloudSegStore] = None
        self.h5_patch_size = int(h5_patch_size)
        self._h5_metas: List[Dict[str, int | str]] = []
        self._h5_prefix: List[int] = []
        self._h5_total_patches = 0
        self._h5_handles: Dict[int, Any] = {}

        mmap_dir = os.path.join(root, "mmap")
        mmap_manifest = os.path.join(mmap_dir, "manifest.json")
        data_dir = os.path.join(root, "data")
        if os.path.isdir(mmap_dir):
            self.mmap_store = MMapCloudSegStore(root)
        elif os.path.isdir(data_dir):
            self.files = sorted(os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".npz"))
            if not self.files:
                raise RuntimeError(f"No .npz files found under {data_dir}")
        elif os.path.isdir(root):
            self.h5_files = sorted(os.path.join(root, name) for name in os.listdir(root) if name.endswith(".h5"))
            if not self.h5_files:
                raise FileNotFoundError(
                    f"Expected either memmap dataset under {mmap_manifest}, npz files under {data_dir}, "
                    f"or .h5 files directly under {root}"
                )
            if h5py is None:
                raise ImportError(
                    "Detected .h5 dataset, but h5py is not installed in the current environment. "
                    "Please install h5py to use CloudSegmentationBig."
                )
            self._build_h5_index()
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
        self.input_channel_indices = None
        if input_channel_indices is not None:
            self.input_channel_indices = [int(channel_idx) for channel_idx in input_channel_indices]
        self.feature_augmentation = dict(feature_augmentation or {})
        gradient_cfg = dict(self.feature_augmentation.get("gradient", {}) or {})
        self.gradient_enabled = bool(gradient_cfg.get("enabled", False))
        self.gradient_skip_channel_indices = {
            int(channel_idx) for channel_idx in gradient_cfg.get("skip_channel_indices", [])
        }
        self.gradient_mode = str(gradient_cfg.get("mode", "magnitude"))
        self.gradient_operator = str(gradient_cfg.get("operator", "sobel"))
        self.gradient_normalizer = ChannelNormalizer(
            NormalizationConfig(**_resolve_gradient_normalization(gradient_cfg.get("normalization")))
        )

    def _select_input_channels(self, image: np.ndarray) -> np.ndarray:
        if self.input_channel_indices is None:
            return image
        return image[self.input_channel_indices]

    def _build_gradient_features(self, image: np.ndarray) -> np.ndarray:
        if not self.gradient_enabled:
            return image

        gradient_features = []
        for channel_idx in range(image.shape[0]):
            if channel_idx in self.gradient_skip_channel_indices:
                continue
            channel = image[channel_idx]
            if self.gradient_operator == "sobel":
                grad_x, grad_y = _sobel_gradients(channel)
            elif self.gradient_operator == "numpy_gradient":
                grad_y, grad_x = np.gradient(channel)
                grad_x = grad_x.astype(np.float32, copy=False)
                grad_y = grad_y.astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unsupported gradient operator: {self.gradient_operator}")

            if self.gradient_mode == "dxdy":
                gradient_features.extend([grad_x, grad_y])
            elif self.gradient_mode == "magnitude":
                gradient = np.sqrt(np.square(grad_x) + np.square(grad_y)).astype(np.float32, copy=False)
                gradient_features.append(gradient)
            else:
                raise ValueError(f"Unsupported gradient feature mode: {self.gradient_mode}")

        if not gradient_features:
            return image

        gradient_array = np.stack(gradient_features, axis=0).astype(np.float32, copy=False)
        gradient_array = self.gradient_normalizer(gradient_array).numpy()
        return np.concatenate([image, gradient_array], axis=0)

    def __len__(self) -> int:
        if self.mmap_store is not None:
            return len(self.mmap_store)
        if self.h5_files:
            return self._h5_total_patches
        return len(self.files)

    def _build_h5_index(self) -> None:
        if self.h5_patch_size <= 0:
            raise ValueError(f"h5_patch_size must be > 0, got {self.h5_patch_size}")

        offset = 0
        for file_path in self.h5_files:
            with h5py.File(file_path, "r") as f:
                image_shape = f["image"].shape
            if len(image_shape) != 3:
                raise ValueError(f"Expected image shape [C,H,W], got {image_shape} in {file_path}")
            _, height, width = image_shape
            rows = int(height) // self.h5_patch_size
            cols = int(width) // self.h5_patch_size
            count = rows * cols
            if count <= 0:
                continue
            self._h5_prefix.append(offset)
            self._h5_metas.append(
                {
                    "file_path": file_path,
                    "rows": rows,
                    "cols": cols,
                    "count": count,
                }
            )
            offset += count

        self._h5_total_patches = offset
        if self._h5_total_patches <= 0:
            raise RuntimeError(f"No valid {self.h5_patch_size}x{self.h5_patch_size} patches found under {len(self.h5_files)} h5 files")

    def _locate_h5_patch(self, index: int) -> Tuple[int, int, int]:
        if index < 0 or index >= self._h5_total_patches:
            raise IndexError(index)
        file_idx = bisect.bisect_right(self._h5_prefix, index) - 1
        local_index = index - self._h5_prefix[file_idx]
        cols = int(self._h5_metas[file_idx]["cols"])
        row_idx = local_index // cols
        col_idx = local_index % cols
        return file_idx, row_idx, col_idx

    def _get_h5_arrays(self, file_idx: int):
        handle = self._h5_handles.get(file_idx)
        if handle is None:
            file_path = str(self._h5_metas[file_idx]["file_path"])
            handle = h5py.File(file_path, "r")
            self._h5_handles[file_idx] = handle
        return handle["image"], handle["label"]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.mmap_store is not None:
            image, mask = self.mmap_store.get(index)
            metadata = self.mmap_store.get_metadata(index)
            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.int64)
        elif self.h5_files:
            file_idx, row_idx, col_idx = self._locate_h5_patch(index)
            top = row_idx * self.h5_patch_size
            left = col_idx * self.h5_patch_size
            image_store, label_store = self._get_h5_arrays(file_idx)
            image = np.asarray(
                image_store[:, top : top + self.h5_patch_size, left : left + self.h5_patch_size],
                dtype=np.float32,
            )
            mask = np.asarray(
                label_store[top : top + self.h5_patch_size, left : left + self.h5_patch_size],
                dtype=np.int64,
            )
            file_path = str(self._h5_metas[file_idx]["file_path"])
            file_stem = os.path.splitext(os.path.basename(file_path))[0]
            sample_id = f"{file_stem}_r{row_idx:02d}_c{col_idx:02d}"
            metadata = {
                "sample_id": sample_id,
                "datetime": parse_sample_datetime(sample_id),
                "source_path": file_path,
            }
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
        image = self.image_normalizer(image).numpy()
        image = self._select_input_channels(image)
        image = self._build_gradient_features(image)
        image = torch.from_numpy(image)
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
    input_channel_indices: Optional[List[int]] = None,
    feature_augmentation: Optional[Dict] = None,
    pad_to_size: Optional[int] = None,
    h5_patch_size: int = 256,
):
    train_ds = CloudSegmentationDataset(
        os.path.join(data_root, train_split),
        ignore_index=ignore_index,
        normalization=normalization,
        input_channel_indices=input_channel_indices,
        feature_augmentation=feature_augmentation,
        h5_patch_size=h5_patch_size,
        transforms=make_transforms(train=True, pad_to_size=pad_to_size, ignore_index=ignore_index),
    )
    val_ds = CloudSegmentationDataset(
        os.path.join(data_root, val_split),
        ignore_index=ignore_index,
        normalization=normalization,
        input_channel_indices=input_channel_indices,
        feature_augmentation=feature_augmentation,
        h5_patch_size=h5_patch_size,
        transforms=make_transforms(train=False, pad_to_size=pad_to_size, ignore_index=ignore_index),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )
    return train_loader, val_loader, train_ds, val_ds
