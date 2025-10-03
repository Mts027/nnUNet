import numpy as np
import pytest
import torch

pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for CCMetrics tests",
)

from CCMetrics import CCDiceMetric, space_separation
from nnunetv2.training.loss.instance_losses import CCMetrics as TorchCCMetrics


def dice_fn(y_pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred_fg = y_pred[1]
    true_fg = y[1]
    intersection = (pred_fg * true_fg).sum()
    denominator = pred_fg.sum() + true_fg.sum()
    if float(denominator) == 0.0:
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    return 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)


def _baseline_score(y_pred_np: np.ndarray, y_np: np.ndarray) -> float:
    cc_dice = CCDiceMetric(cc_reduction="patient")
    for batch_index in range(y_np.shape[0]):
        space_separation.compute_voronoi_regions(
            y_np[batch_index, 1].astype(np.uint16, copy=False)
        )
        # CCDiceMetric from CC-Metrics only supports batch size 1, so feed samples separately.
        cc_dice(
            y_pred=y_pred_np[batch_index : batch_index + 1],
            y=y_np[batch_index : batch_index + 1],
        )
    score = (1.0 - cc_dice.cc_aggregate()).mean()
    return float(score)


@pytest.mark.parametrize("spatial", [(24, 24, 24)])
def test_ccmetrics_matches_ccdice_empty(spatial):
    batch = 1
    y_pred_np = np.zeros((batch, 2, *spatial), dtype=np.float32)
    y_pred_np[:, 0] = 1.0
    y_np = y_pred_np.copy()

    baseline = _baseline_score(y_pred_np, y_np)

    device = torch.device("cuda")
    module = TorchCCMetrics(metric=dice_fn, activation=None).to(device)
    y_pred_t = torch.from_numpy(y_pred_np).to(device)
    y_idx = torch.zeros((batch, 1, *spatial), dtype=torch.int64, device=device)

    torch_score = module(y_pred_t, y_idx).detach().cpu().item()

    assert torch_score == pytest.approx(baseline, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize("spatial", [(24, 24, 24)])
def test_ccmetrics_matches_ccdice_gt_nonempty_pred_empty(spatial):
    batch = 1
    center = torch.tensor([s // 2 for s in spatial]).float()
    coords = torch.stack(torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32) for s in spatial], indexing="ij"
    ), dim=0)
    radius = min(spatial) // 4
    squared_dist = ((coords - center.view(3, 1, 1, 1)) ** 2).sum(0)
    mask = (squared_dist <= radius ** 2).float()

    y_np = np.zeros((batch, 2, *spatial), dtype=np.float32)
    y_np[:, 1] = mask.numpy()
    y_np[:, 0] = 1.0 - y_np[:, 1]

    y_pred_np = np.zeros_like(y_np)
    y_pred_np[:, 0] = 1.0

    baseline = _baseline_score(y_pred_np, y_np)

    device = torch.device("cuda")
    module = TorchCCMetrics(metric=dice_fn, activation=None).to(device)
    y_pred_t = torch.from_numpy(y_pred_np).to(device)
    y_idx = torch.from_numpy(np.argmax(y_np, axis=1, keepdims=True).astype(np.int64)).to(device)

    torch_score = module(y_pred_t, y_idx).detach().cpu().item()

    assert torch_score == pytest.approx(baseline, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize(
    "spatial,p_fg_gt,p_fg_pred,seed",
    [
        ((16, 16, 16), 0.5, 0.5, 0),
        ((12, 18, 20), 0.2, 0.7, 1),
        ((20, 20, 12), 0.8, 0.3, 2),
    ],
)
def test_ccmetrics_matches_ccdice_random_batch(spatial, p_fg_gt, p_fg_pred, seed):
    batch = 4
    rng = np.random.default_rng(seed)

    gt_fg = (rng.random((batch, *spatial), dtype=np.float32) < p_fg_gt).astype(np.float32)
    pred_fg = (rng.random((batch, *spatial), dtype=np.float32) < p_fg_pred).astype(np.float32)

    gt_np = np.stack([1.0 - gt_fg, gt_fg], axis=1)
    pred_np = np.stack([1.0 - pred_fg, pred_fg], axis=1)

    cc_dice = CCDiceMetric(cc_reduction="patient")
    for batch_index in range(batch):
        space_separation.compute_voronoi_regions(
            gt_np[batch_index, 1].astype(np.uint16, copy=False)
        )
        cc_dice(
            y_pred=pred_np[batch_index : batch_index + 1],
            y=gt_np[batch_index : batch_index + 1],
        )
    baseline = float((1.0 - cc_dice.cc_aggregate()).mean())

    device = torch.device("cuda")
    batch_pred = torch.from_numpy(pred_np).to(device)
    batch_idx = torch.from_numpy(np.argmax(gt_np, axis=1, keepdims=True).astype(np.int64)).to(device)

    module = TorchCCMetrics(metric=dice_fn, activation=None).to(device)
    torch_score = module(batch_pred, batch_idx).detach().cpu().item()

    assert torch_score == pytest.approx(baseline, rel=0.01, abs=1e-2)
