from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.instance_losses import BlobLoss, CCMetrics
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.git_logging import log_git_context
from nnunetv2.utilities.helpers import softmax_helper_dim1


class LossMixer(nn.Module):
    """Combine multiple loss modules into a weighted sum.

    Notes
    -----
    * Components are stored in a :class:`torch.nn.ModuleDict` so that they are moved
      correctly when ``LossMixer.to(device)`` is called.
    * The ``forward`` call accepts the default nnU-Net loss signature ``(y_pred, y)``
      and forwards it to every registered loss module.
    """

    def __init__(self, components: Sequence[Tuple[str, nn.Module, float]]):
        super().__init__()
        if not components:
            raise ValueError("LossMixer requires at least one component")

        names = [name for name, _, _ in components]
        if len(set(names)) != len(names):
            raise ValueError("LossMixer component names must be unique")

        self.components = nn.ModuleDict({name: module for name, module, _ in components})
        self.weights = {name: float(weight) for name, _, weight in components}
        self._last_components: dict[str, torch.Tensor] = {}

    @property
    def component_losses(self) -> Mapping[str, torch.Tensor]:
        """Return the losses computed in the previous forward pass."""

        return self._last_components

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        total = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        self._last_components = {}
        for name, module in self.components.items():
            weight = self.weights[name]
            if weight == 0:
                continue
            value = module(y_pred, y)
            # store detached copy for logging/debugging without keeping graph
            self._last_components[name] = value.detach()
            total = total + weight * value
        return total


def integrate_deep_supervision(trainer: nnUNetTrainer, loss: nn.Module) -> nn.Module:
    """Wrap a loss with nnU-Net's deep supervision helper if required."""

    if not trainer.enable_deep_supervision:
        return loss

    deep_supervision_scales = trainer._get_deep_supervision_scales()
    weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
    if trainer.is_ddp and not trainer._do_i_compile():
        weights[-1] = 1e-6
    else:
        weights[-1] = 0

    weights /= weights.sum()
    return DeepSupervisionWrapper(loss, weights)


def _safe_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_sum = mask.sum()
    if mask_sum <= 0:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return (value * mask).sum() / mask_sum


def _component_dice_ce_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-5
    mask_union = (pred_mask | true_mask)
    mask_float = mask_union.to(dtype=y_pred.dtype)

    foreground_prob = torch.clamp(y_pred[1], min=eps, max=1.0 - eps)
    foreground_true = y_true[1]

    masked_pred = foreground_prob * mask_float
    masked_true = foreground_true * mask_float

    intersection = (masked_pred * masked_true).sum()
    denominator = masked_pred.sum() + masked_true.sum()
    if float(denominator) > 0.0:
        dice_loss = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    else:
        dice_loss = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

    log_probs = torch.log(torch.clamp(y_pred, min=eps, max=1.0 - eps))
    per_voxel_ce = -(y_true * log_probs).sum(dim=0)
    ce_loss = _safe_mean(per_voxel_ce, mask_union)

    return dice_loss + ce_loss


def _build_global_dc_ce_loss(trainer: nnUNetTrainer) -> nn.Module:
    loss = DC_and_CE_loss(
        {
            'batch_dice': trainer.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': trainer.is_ddp,
        },
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=trainer.label_manager.ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )

    if hasattr(loss, "dc") and trainer._do_i_compile():
        loss.dc = torch.compile(loss.dc)  # type: ignore[attr-defined]
    return loss


def _assert_instance_loss_prerequisites(trainer: nnUNetTrainer):
    if trainer.label_manager.has_regions:
        raise AssertionError("Instance-level losses require label-based training without regions")
    if trainer.label_manager.has_ignore_label:
        raise AssertionError("Instance-level losses do not support ignore labels")
    if trainer.label_manager.num_segmentation_heads != 2:
        raise AssertionError("Blob/CC instance losses require a binary segmentation (background + foreground)")


class _InstanceTrainerBase(nnUNetTrainer):
    include_global_component: bool = False
    global_component_weight: float = 1.0
    instance_component_weight: float = 1.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        log_git_context(self)

    def _build_instance_loss(self) -> nn.Module:
        raise NotImplementedError

    def _build_loss(self) -> nn.Module:  # type: ignore[override]
        _assert_instance_loss_prerequisites(self)

        instance_loss = self._build_instance_loss().to(self.device)

        if not self.include_global_component:
            return integrate_deep_supervision(self, instance_loss)

        global_loss = _build_global_dc_ce_loss(self).to(self.device)
        components = [
            ("global", global_loss, self.global_component_weight),
            ("instance", instance_loss, self.instance_component_weight),
        ]
        mixed_loss = LossMixer(components)
        return integrate_deep_supervision(self, mixed_loss)


class nnUNetTrainerCCDiceCE(_InstanceTrainerBase):
    def _build_instance_loss(self) -> nn.Module:
        cc_loss = CCMetrics(metric=_component_dice_ce_loss, activation=softmax_helper_dim1)
        return cc_loss


class nnUNetTrainerBlobDiceCE(_InstanceTrainerBase):
    def _build_instance_loss(self) -> nn.Module:
        blob_loss = BlobLoss(metric=_component_dice_ce_loss, activation=softmax_helper_dim1)
        return blob_loss


class nnUNetTrainerGlobalCCDiceCE(_InstanceTrainerBase):
    include_global_component = True

    def _build_instance_loss(self) -> nn.Module:
        cc_loss = CCMetrics(metric=_component_dice_ce_loss, activation=softmax_helper_dim1)
        return cc_loss


class nnUNetTrainerGlobalBlobDiceCE(_InstanceTrainerBase):
    include_global_component = True

    def _build_instance_loss(self) -> nn.Module:
        blob_loss = BlobLoss(metric=_component_dice_ce_loss, activation=softmax_helper_dim1)
        return blob_loss
