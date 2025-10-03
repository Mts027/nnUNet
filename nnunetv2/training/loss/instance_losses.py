import torch
import torch.nn.functional as F
from typing import Callable, Optional
from nnunetv2.utilities.connected_components import get_voronoi, get_cc

class CCMetrics(torch.nn.Module):
    def __init__(self, metric, activation) -> None:
        super().__init__()

        self.metric = metric
        self.activation = activation

    def forward(self, y_pred, y):
        assert y_pred.ndim == 5 and y_pred.shape[1] == 2, f"Expected y_pred with shape [B,2,H,W,D], but got {tuple(y_pred.shape)}"
        assert list(y.shape) == [y_pred.shape[0], 1, *y_pred.shape[2:]], f"Expected y with shape ({tuple(y_pred.shape)}) [B,1,H,W,D], but got {tuple(y.shape)}"
        assert y.dtype == torch.int64, f"Expected y.dtype=torch.int64, but got {y.dtype}"

        assert y_pred.is_cuda, f"CCMetrics expects CUDA tensors for y_pred, but got device {y_pred.device}."
        assert y.is_cuda, f"CCMetrics expects CUDA tensors for y, but got device {y.device}."
        assert y.device == y_pred.device, f"y and y_pred must reside on the same CUDA device, but got y on {y.device} and y_pred on {y_pred.device}."

        y_idx = y[:, 0].long()  # [B,*]
        y = F.one_hot(y_idx, num_classes=y_pred.shape[1]).movedim(-1, 1).float()  # [B,2,*]

        voronoi = get_voronoi(y, do_bg=False)
        voronoi = voronoi[:, 0, ...]

        assert y_pred.shape == y.shape, "y_pred and one-hot y must match"
        assert voronoi.dtype == torch.int64
        assert list(voronoi.shape) == [y_pred.shape[0], *y_pred.shape[2:]], f"Expected voronoi with shape [B,H,W,D], but got {tuple(voronoi.shape)}"

        def cc_metrics_masking(component_id: int, voronoi: torch.Tensor):
            mask: torch.Tensor = voronoi == component_id
            return mask, mask
 
        result = binary_cc(
            y_pred=y_pred,
            y=y,
            voronoi=voronoi,
            metric=self.metric,
            masking_fn=cc_metrics_masking,
            activation=self.activation,
        )
        return result

class BlobLoss(torch.nn.Module):
    def __init__(self, metric, activation) -> None:
        super().__init__()

        self.metric = metric
        self.activation = activation

    def forward(self, y_pred, y):
        assert y_pred.ndim == 5 and y_pred.shape[1] == 2, f"Expected y_pred with shape [B,2,H,W,D], but got {tuple(y_pred.shape)}"
        assert list(y.shape) == [y_pred.shape[0], 1, *y_pred.shape[2:]], f"Expected y with shape ({tuple(y_pred.shape)}) [B,1,H,W,D], but got {tuple(y.shape)}"
        assert y.dtype == torch.int64, f"Expected y.dtype=torch.int64, but got {y.dtype}"

        assert y_pred.is_cuda, f"BlobLoss expects CUDA tensors for y_pred, but got device {y_pred.device}."
        assert y.is_cuda, f"BlobLoss expects CUDA tensors for y, but got device {y.device}."
        assert y.device == y_pred.device, f"y and y_pred must reside on the same CUDA device, but got y on {y.device} and y_pred on {y_pred.device}."

        y_idx = y[:, 0].long()  # [B,*]
        y = F.one_hot(y_idx, num_classes=y_pred.shape[1]).movedim(-1, 1).float()  # [B,2,*]

        connected_components = get_cc(y, do_bg=False)
        connected_components = connected_components[:, 0, ...]

        assert y_pred.shape == y.shape, "y_pred and one-hot y must match"
        assert connected_components.dtype == torch.int64
        assert list(connected_components.shape) == [y_pred.shape[0], *y_pred.shape[2:]], f"Expected voronoi with shape [B,H,W,D], but got {tuple(voronoi.shape)}"

        def cc_metrics_masking(component_id: int, connected_components: torch.Tensor):
            pred_mask = (connected_components == component_id) | (connected_components == 0)
            true_mask = (connected_components == component_id)
            return pred_mask, true_mask 

        result = binary_cc(
            y_pred=y_pred,
            y=y,
            voronoi=connected_components,
            metric=self.metric,
            masking_fn=cc_metrics_masking,
            activation=self.activation,
        )
        return result


def per_channel_cc(
    y_pred: torch.Tensor,  # [*spatial]
    y: torch.Tensor,  # [*spatial]
    cc: torch.Tensor,  # [*spatial], integer component IDs
    metric: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    masking_fn: Callable[[int, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Compute metric on each connected component in one channel,
    but wrap each component's computation in a checkpoint to reduce memory.
    """
    max_id = cc.max()
    if max_id == 0:
        # fallback: treat whole channel as one component
        return metric(y_pred, y, torch.ones_like(y), torch.ones_like(y))

    ids = torch.unique(cc)
    ids = ids[ids > 0]

    # define pure function for checkpointing
    def _component_score(pred: torch.Tensor, true: torch.Tensor, pred_mask: torch.Tensor, true_mask: torch.Tensor):
        assert pred.shape == true.shape
        assert true.ndim == 4
        assert true.shape[0] == 2

        assert pred_mask.ndim == 3
        assert list(pred_mask.shape) == list(true.shape)[1:]
        assert true_mask.ndim == 3
        assert list(true_mask.shape) == list(true.shape)[1:]

        assert pred_mask.dtype == torch.bool
        assert true_mask.dtype == torch.bool
        
        masked_pred = pred * pred_mask
        masked_true = true * true_mask

        return metric(masked_pred, masked_true, pred_mask, true_mask)

    scores: list[torch.Tensor] = []
    for comp_id in ids:
        pred_mask, true_mask = masking_fn(comp_id, cc)
        # checkpoint the component-level forward
        score: torch.Tensor = torch.utils.checkpoint.checkpoint(_component_score, y_pred, y, pred_mask, true_mask, use_reentrant = False)
        assert score.ndim == 0, "metric_fn should return scalar functions"
        scores.append(score)

    return torch.stack(scores, dim=0).mean()


def binary_cc(
    y_pred: torch.Tensor,  # [B, C, *spatial]
    y: torch.Tensor,  # [B, C, *spatial]
    voronoi: torch.Tensor,  # [B, C - 1 if not include_first_channel, *spatial], integer CC labels
    metric: Callable,  # fn(y_pred[N,...], y[N,...]) -> Tensor of shape [N]
    masking_fn: Callable[[int, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    activation: Optional[Callable],
) -> torch.Tensor:
    """Reduce a metric over connected components and channels to obtain a batch score.

    Args:
        y_pred: Logits or probabilities shaped ``[B, C, H, W, D]``.
        y: One-hot ground truth tensor with the same shape as ``y_pred``.
        voronoi: Integer labels describing connected components per channel.
        metric: Callable applied to each component; should accept ``(pred, true)`` tensors.
        activation: Optional callable applied to ``y_pred`` before metric evaluation.
    """
    # -- sanity checks --
    assert y_pred.shape == y.shape, "All inputs must match in shape"
    assert y.ndim == 5, "Expect [B, C, H, W, D]"
    assert y.shape[1] == 2, "This is for binary inputs only"

    assert y.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert y_pred.dtype in (torch.float16, torch.bfloat16, torch.float32)

    # -- optional activation --
    if activation is not None:
        y_pred = activation(y_pred)

    B = y_pred.shape[0]
    sample_scores: list[torch.Tensor] = []

    # Instead of vmap, loop over the batch
    for i in range(B):
        score = per_channel_cc(y_pred[i], y[i], voronoi[i], metric, masking_fn)
        sample_scores.append(score)

    # Stack the results into a tensor (if that's what vmap was producing)
    scores = torch.stack(sample_scores, dim=0)

    return scores.mean()
