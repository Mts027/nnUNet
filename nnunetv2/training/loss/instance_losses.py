import torch
import torch.nn.functional as F
from typing import Callable, Optional

class CCBase(torch.nn.Module):
    """Wrap a metric so it can operate on connected-component partitions."""

    def __init__(self, metric, has_regions, activation) -> None:
        super().__init__()

        self.metric = metric
        # Not used anymore - we will keep it for api compatibility
        self.has_regions = has_regions
        self.activation = activation

    # @torch.compile
    def forward(self, y_pred, y):
        skip_first = True
        assert y_pred.ndim == 5 and y_pred.shape[1] > 1, "Expect [B,>1,H,W,D] logits"
        assert y.shape[1] == 1

        y_idx = y[:, 0].long()  # [B,*]
        y = F.one_hot(y_idx, num_classes=y_pred.shape[1]).movedim(-1, 1).float()  # [B,2,*]

        if skip_first:
            y_pred = y_pred[:, 1:, ...]
            y = y[:, 1:, ...]

        cc_assignments = get_voronoi(y, do_bg=skip_first, use_cpu=False)

        assert y_pred.shape == y.shape, "y_pred and one-hot y must match"
        assert cc_assignments.shape == y_pred.shape
        assert cc_assignments.dtype == torch.int64
 
        return cc(
            y_pred=y_pred,
            y=y,
            voronoi=cc_assignments,
            metric=self.metric,
            channel_reduction_fn=torch.mean,
            include_first_channel=skip_first,
            activation=self.activation,
        )


def per_channel_cc(
    y_pred: torch.Tensor,  # [*spatial]
    y: torch.Tensor,  # [*spatial]
    cc: torch.Tensor,  # [*spatial], integer component IDs
    metric: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute metric on each connected component in one channel,
    but wrap each component's computation in a checkpoint to reduce memory.
    """
    max_id = cc.max()
    if max_id == 0:
        # fallback: treat whole channel as one component
        return metric(y_pred, y, torch.ones_likes(y))

    ids = torch.unique(cc)
    ids = ids[ids > 0]

    # define pure function for checkpointing
    def _component_score(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor):
        assert pred.shape == true.shape
        assert true.ndim == 4
        assert true.shape[0] == 2
        assert mask.ndim == 3
        assert list(mask.shape) == list(true.shape)[1:]
        
        masked_pred = pred * mask
        masked_true = true * mask

        return metric(masked_pred, masked_true, mask)

    scores: list[torch.Tensor] = []
    for comp_id in ids:
        mask = (cc == comp_id).to(y_pred.dtype)
        # checkpoint the component-level forward
        score: torch.Tensor = torch.utils.checkpoint(_component_score, y_pred, y, mask, use_reentrant = False)
        assert score.ndim == 0
        scores.append(score)
    return torch.stack(scores, dim=0).mean()


def binary_cc(
    y_pred: torch.Tensor,  # [B, C, *spatial]
    y: torch.Tensor,  # [B, C, *spatial]
    voronoi: torch.Tensor,  # [B, C - 1 if not include_first_channel, *spatial], integer CC labels
    metric: Callable,  # fn(y_pred[N,...], y[N,...]) -> Tensor of shape [N]
    activation: Optional[Callable],
) -> torch.Tensor:
    """Reduce a metric over connected components and channels to obtain a batch score.

    Args:
        y_pred: Logits or probabilities shaped ``[B, C, H, W, D]``.
        y: One-hot ground truth tensor with the same shape as ``y_pred``.
        voronoi: Integer labels describing connected components per channel.
        metric: Callable applied to each component; should accept ``(pred, true)`` tensors.
        channel_reduction_fn: Aggregation applied after component scores per channel.
        include_first_channel: Whether channel 0 is part of the reduction.
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

    B, C = y_pred.shape[:2]
    sample_scores: list[torch.Tensor] = []

    scores = torch.func.vmap(per_channel_cc, in_dims=(0, 0, 0, None,))(y_pred, y, voronoi, metric)

    return scores.mean()


