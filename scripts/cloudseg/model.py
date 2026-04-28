from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.generator import LightningDiTBlock, TorchLinear, sincos_init


def _resize_bhwc(x: jax.Array, height: int, width: int) -> jax.Array:
    return jax.image.resize(x, (x.shape[0], height, width, x.shape[-1]), method="bilinear")


class DriftingBackbone(nn.Module):
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 32
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    use_qknorm: bool = False
    use_swiglu: bool = False
    use_rope: bool = False
    use_rmsnorm: bool = False
    cond_dim: int | None = None
    n_cls_tokens: int = 0
    attn_fp32: bool = True
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_remat: bool = False

    @nn.compact
    def __call__(self, x, c, deterministic=True):
        batch, height, width, channels = x.shape
        patch_size = self.patch_size
        target_grid = self.input_size // patch_size
        num_patches = target_grid * target_grid
        effective_patch = height // target_grid
        grid_h, grid_w = target_grid, target_grid

        x = x.reshape(batch, grid_h, effective_patch, grid_w, effective_patch, channels)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape(batch, num_patches, effective_patch * effective_patch * channels)
        x = TorchLinear(self.hidden_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        pos_embed = self.param(
            "pos_embed",
            sincos_init(self.hidden_size, num_patches),
            (1, num_patches, self.hidden_size),
        )
        x = (x + pos_embed).astype(self.dtype)

        if self.n_cls_tokens > 0:
            c = c.astype(self.dtype)
            c_tokens = TorchLinear(self.hidden_size, bias=True, dtype=self.dtype, param_dtype=self.param_dtype)(c)
            c_tokens = jnp.expand_dims(c_tokens, 1)
            c_tokens = jnp.tile(c_tokens, (1, self.n_cls_tokens, 1))
            cls_embed = self.param(
                "cls_embed",
                nn.initializers.normal(stddev=0.02),
                (1, self.n_cls_tokens, self.hidden_size),
            )
            c_tokens = c_tokens + cls_embed
            x = jnp.concatenate([c_tokens, x], axis=1)
            x = x.astype(self.dtype)

        block_cls = LightningDiTBlock
        if self.use_remat:
            block_cls = nn.remat(LightningDiTBlock, prevent_cse=True)

        for i in range(self.depth):
            x = block_cls(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_qknorm=self.use_qknorm,
                use_swiglu=self.use_swiglu,
                use_rmsnorm=self.use_rmsnorm,
                cond_dim=self.cond_dim,
                use_rope=self.use_rope,
                attn_fp32=self.attn_fp32,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"blocks_{i}",
            )(x, c, deterministic)

        if self.n_cls_tokens > 0:
            x = x[:, self.n_cls_tokens :, :]
        x = x.reshape(batch, grid_h, grid_w, self.hidden_size)
        return x


class InputAdapter(nn.Module):
    out_channels: int
    hidden_channels: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(self.out_channels, kernel_size=(1, 1), padding="SAME")(x)
        return x


class SegmentationHead(nn.Module):
    num_classes: int
    hidden_channels: int = 256

    @nn.compact
    def __call__(self, x, out_height: int, out_width: int):
        x = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(self.hidden_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.gelu(x)
        x = _resize_bhwc(x, out_height, out_width)
        return nn.Conv(self.num_classes, kernel_size=(1, 1), padding="SAME")(x)


class CloudSegAdapter(nn.Module):
    backbone_cfg: Dict[str, Any]
    num_classes: int = 10
    input_channels: int = 17
    adapter_hidden_channels: int = 64
    head_hidden_channels: int = 256
    use_input_adapter: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):
        batch, height, width, _ = x.shape
        cond_dim = int(self.backbone_cfg["cond_dim"])
        backbone_input_size = int(self.backbone_cfg["input_size"])

        if self.use_input_adapter:
            x = InputAdapter(
                out_channels=int(self.backbone_cfg["in_channels"]),
                hidden_channels=self.adapter_hidden_channels,
            )(x)
        else:
            expected_channels = int(self.backbone_cfg["in_channels"])
            if x.shape[-1] != expected_channels:
                raise ValueError(
                    f"Input channels ({x.shape[-1]}) must match backbone.in_channels ({expected_channels}) "
                    "when use_input_adapter is false."
                )
        if x.shape[1] != backbone_input_size or x.shape[2] != backbone_input_size:
            x = _resize_bhwc(x, backbone_input_size, backbone_input_size)

        cond_embed = self.param("cond_embed", nn.initializers.zeros, (cond_dim,))
        cond = jnp.tile(cond_embed[None, :], (batch, 1))

        features = DriftingBackbone(
            **self.backbone_cfg,
            name="LightningDiT_0",
        )(x, cond, deterministic=deterministic)
        logits = SegmentationHead(
            num_classes=self.num_classes,
            hidden_channels=self.head_hidden_channels,
        )(features, out_height=height, out_width=width)
        return logits


class _ConvBNReLU(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        x = nn.relu(x)
        return x


class _DoubleConv(nn.Module):
    channels1: int
    channels2: int
    channels3: int | None = None

    @nn.compact
    def __call__(self, x, train: bool):
        x = _ConvBNReLU(self.channels1)(x, train=train)
        x = _ConvBNReLU(self.channels2)(x, train=train)
        if self.channels3 is not None:
            x = _ConvBNReLU(self.channels3)(x, train=train)
        return x


class OfficialUNet(nn.Module):
    input_channels: int = 17
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        train = not deterministic

        stage_1 = _DoubleConv(32, 64, 64, name="stage_1")(x, train=train)
        stage_2 = nn.max_pool(stage_1, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_2 = _DoubleConv(128, 128, name="stage_2")(stage_2, train=train)

        stage_3 = nn.max_pool(stage_2, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_3 = _DoubleConv(256, 256, name="stage_3")(stage_3, train=train)

        stage_4 = nn.max_pool(stage_3, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_4 = _DoubleConv(512, 512, name="stage_4")(stage_4, train=train)

        stage_5 = nn.max_pool(stage_4, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_5 = _DoubleConv(1024, 1024, name="stage_5")(stage_5, train=train)

        up_4 = nn.ConvTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_4")(stage_5)
        up_4 = jnp.concatenate([up_4, stage_4], axis=-1)
        up_4 = _DoubleConv(512, 512, name="stage_up_4")(up_4, train=train)

        up_3 = nn.ConvTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_3")(up_4)
        up_3 = jnp.concatenate([up_3, stage_3], axis=-1)
        up_3 = _DoubleConv(256, 256, name="stage_up_3")(up_3, train=train)

        up_2 = nn.ConvTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_2")(up_3)
        up_2 = jnp.concatenate([up_2, stage_2], axis=-1)
        up_2 = _DoubleConv(128, 128, name="stage_up_2")(up_2, train=train)

        up_1 = nn.ConvTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_1")(up_2)
        up_1 = jnp.concatenate([up_1, stage_1], axis=-1)
        up_1 = _DoubleConv(64, 64, name="stage_up_1")(up_1, train=train)

        logits = nn.Conv(self.num_classes, kernel_size=(3, 3), padding="SAME", name="final")(up_1)
        return logits


class OfficialUNetHierarchical(nn.Module):
    input_channels: int = 17
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        train = not deterministic

        stage_1 = _DoubleConv(32, 64, 64, name="stage_1")(x, train=train)
        stage_2 = nn.max_pool(stage_1, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_2 = _DoubleConv(128, 128, name="stage_2")(stage_2, train=train)

        stage_3 = nn.max_pool(stage_2, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_3 = _DoubleConv(256, 256, name="stage_3")(stage_3, train=train)

        stage_4 = nn.max_pool(stage_3, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_4 = _DoubleConv(512, 512, name="stage_4")(stage_4, train=train)

        stage_5 = nn.max_pool(stage_4, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_5 = _DoubleConv(1024, 1024, name="stage_5")(stage_5, train=train)

        up_4 = nn.ConvTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_4")(stage_5)
        up_4 = jnp.concatenate([up_4, stage_4], axis=-1)
        up_4 = _DoubleConv(512, 512, name="stage_up_4")(up_4, train=train)

        up_3 = nn.ConvTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_3")(up_4)
        up_3 = jnp.concatenate([up_3, stage_3], axis=-1)
        up_3 = _DoubleConv(256, 256, name="stage_up_3")(up_3, train=train)

        up_2 = nn.ConvTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_2")(up_3)
        up_2 = jnp.concatenate([up_2, stage_2], axis=-1)
        up_2 = _DoubleConv(128, 128, name="stage_up_2")(up_2, train=train)

        up_1 = nn.ConvTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_1")(up_2)
        up_1 = jnp.concatenate([up_1, stage_1], axis=-1)
        up_1 = _DoubleConv(64, 64, name="stage_up_1")(up_1, train=train)

        # Level-1: clear/cloud (2 classes)
        logits_l1 = nn.Conv(2, kernel_size=(3, 3), padding="SAME", name="head_l1")(up_1)
        # Level-2 under cloud: high/middle/low/thick_or_vertical (4 groups)
        logits_l2_cloud = nn.Conv(4, kernel_size=(3, 3), padding="SAME", name="head_l2_cloud")(up_1)
        # Level-3 conditional heads per group
        logits_high = nn.Conv(2, kernel_size=(3, 3), padding="SAME", name="head_l3_high")(up_1)  # Ci/Cs
        logits_middle = nn.Conv(2, kernel_size=(3, 3), padding="SAME", name="head_l3_middle")(up_1)  # Ac/As
        logits_low = nn.Conv(3, kernel_size=(3, 3), padding="SAME", name="head_l3_low")(up_1)  # Cu/Sc/St
        logits_thick = nn.Conv(2, kernel_size=(3, 3), padding="SAME", name="head_l3_thick")(up_1)  # DC/Ns

        log_p_l1 = jax.nn.log_softmax(logits_l1, axis=-1)
        log_p_clear = log_p_l1[..., 0]
        log_p_cloud = log_p_l1[..., 1]

        log_p_l2_cloud = jax.nn.log_softmax(logits_l2_cloud, axis=-1)  # [high, middle, low, thick]
        log_p_high = log_p_cloud + log_p_l2_cloud[..., 0]
        log_p_middle = log_p_cloud + log_p_l2_cloud[..., 1]
        log_p_low = log_p_cloud + log_p_l2_cloud[..., 2]
        log_p_thick = log_p_cloud + log_p_l2_cloud[..., 3]

        log_p_high_cls = jax.nn.log_softmax(logits_high, axis=-1)
        log_p_middle_cls = jax.nn.log_softmax(logits_middle, axis=-1)
        log_p_low_cls = jax.nn.log_softmax(logits_low, axis=-1)
        log_p_thick_cls = jax.nn.log_softmax(logits_thick, axis=-1)

        # Final 10-class order:
        # 0:Clr, 1:Ci, 2:Cs, 3:DC, 4:Ac, 5:As, 6:Ns, 7:Cu, 8:Sc, 9:St
        logits_l3 = jnp.stack(
            [
                log_p_clear,
                log_p_high + log_p_high_cls[..., 0],      # Ci
                log_p_high + log_p_high_cls[..., 1],      # Cs
                log_p_thick + log_p_thick_cls[..., 0],    # DC
                log_p_middle + log_p_middle_cls[..., 0],  # Ac
                log_p_middle + log_p_middle_cls[..., 1],  # As
                log_p_thick + log_p_thick_cls[..., 1],    # Ns
                log_p_low + log_p_low_cls[..., 0],        # Cu
                log_p_low + log_p_low_cls[..., 1],        # Sc
                log_p_low + log_p_low_cls[..., 2],        # St
            ],
            axis=-1,
        )

        logits_l2 = jnp.stack(
            [
                log_p_clear,
                log_p_high,
                log_p_middle,
                log_p_low,
                log_p_thick,
            ],
            axis=-1,
        )
        return {
            "logits_l1": logits_l1,
            "logits_l2": logits_l2,
            "logits_l3": logits_l3,
        }


class DriftingUNet(nn.Module):
    input_channels: int = 17
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, deterministic: bool = True, *, return_features: bool = False):
        train = not deterministic

        stage_1 = _DoubleConv(32, 64, 64, name="stage_1")(x, train=train)
        stage_2_in = nn.max_pool(stage_1, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_2 = _DoubleConv(128, 128, name="stage_2")(stage_2_in, train=train)

        stage_3_in = nn.max_pool(stage_2, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_3 = _DoubleConv(256, 256, name="stage_3")(stage_3_in, train=train)

        stage_4_in = nn.max_pool(stage_3, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        stage_4 = _DoubleConv(512, 512, name="stage_4")(stage_4_in, train=train)

        bottleneck_in = nn.max_pool(stage_4, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        bottleneck = _DoubleConv(1024, 1024, name="stage_5")(bottleneck_in, train=train)

        up_4 = nn.ConvTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_4")(bottleneck)
        up_4 = jnp.concatenate([up_4, stage_4], axis=-1)
        up_4 = _DoubleConv(512, 512, name="stage_up_4")(up_4, train=train)

        up_3 = nn.ConvTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_3")(up_4)
        up_3 = jnp.concatenate([up_3, stage_3], axis=-1)
        up_3 = _DoubleConv(256, 256, name="stage_up_3")(up_3, train=train)

        up_2 = nn.ConvTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_2")(up_3)
        up_2 = jnp.concatenate([up_2, stage_2], axis=-1)
        up_2 = _DoubleConv(128, 128, name="stage_up_2")(up_2, train=train)

        up_1 = nn.ConvTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="upsample_1")(up_2)
        up_1 = jnp.concatenate([up_1, stage_1], axis=-1)
        up_1 = _DoubleConv(64, 64, name="stage_up_1")(up_1, train=train)

        logits = nn.Conv(self.num_classes, kernel_size=(3, 3), padding="SAME", name="final")(up_1)

        if not return_features:
            return logits

        return {
            "logits": logits,
            "features": {
                "enc2": stage_2,
                "enc3": stage_3,
                "enc4": stage_4,
                "bottleneck": bottleneck,
                "dec3": up_3,
                "dec2": up_2,
                "dec1": up_1,
            },
        }


def extract_backbone_config(metadata: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(metadata.get("model_config", {}) or {})
    use_bf16 = bool(model_cfg.get("use_bf16", False))
    dtype = jnp.bfloat16 if use_bf16 else jnp.float32
    return {
        "input_size": int(model_cfg["input_size"]),
        "patch_size": int(model_cfg["patch_size"]),
        "in_channels": int(model_cfg["in_channels"]),
        "hidden_size": int(model_cfg["hidden_size"]),
        "depth": int(model_cfg["depth"]),
        "num_heads": int(model_cfg["num_heads"]),
        "mlp_ratio": float(model_cfg["mlp_ratio"]),
        "use_qknorm": bool(model_cfg.get("use_qknorm", False)),
        "use_swiglu": bool(model_cfg.get("use_swiglu", False)),
        "use_rope": bool(model_cfg.get("use_rope", False)),
        "use_rmsnorm": bool(model_cfg.get("use_rmsnorm", False)),
        "cond_dim": int(model_cfg["cond_dim"]),
        "n_cls_tokens": int(model_cfg.get("n_cls_tokens", 0)),
        "attn_fp32": bool(model_cfg.get("attn_fp32", True)),
        "dtype": dtype,
        "param_dtype": jnp.float32,
        "use_remat": bool(model_cfg.get("use_remat", False)),
    }


def default_backbone_config() -> Dict[str, Any]:
    return {
        "input_size": 256,
        "patch_size": 16,
        "in_channels": 3,
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
        "cond_dim": 768,
        "n_cls_tokens": 16,
        "attn_fp32": True,
        "dtype": jnp.bfloat16,
        "param_dtype": jnp.float32,
        "use_remat": True,
    }
