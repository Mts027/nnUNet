import numpy as np
import pytest
import torch

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
    cc_dice(y_pred=y_pred_np, y=y_np)
    score = (1.0 - cc_dice.cc_aggregate()).mean()
    return float(score)


@pytest.mark.parametrize("spatial", [(24, 24, 24)])
def test_ccmetrics_matches_ccdice_empty(spatial):
    batch = 1
    y_pred_np = np.zeros((batch, 2, *spatial), dtype=np.float32)
    y_pred_np[:, 0] = 1.0
    y_np = y_pred_np.copy()

    baseline = _baseline_score(y_pred_np, y_np)

    module = TorchCCMetrics(metric=dice_fn, activation=None)
    y_pred_t = torch.from_numpy(y_pred_np)
    y_idx = torch.zeros((batch, 1, *spatial), dtype=torch.int64)

    torch_score = module(y_pred_t, y_idx).item()

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

    module = TorchCCMetrics(metric=dice_fn, activation=None)
    y_pred_t = torch.from_numpy(y_pred_np)
    y_idx = torch.from_numpy(np.argmax(y_np, axis=1, keepdims=True).astype(np.int64))

    torch_score = module(y_pred_t, y_idx).item()

    assert torch_score == pytest.approx(baseline, rel=1e-4, abs=1e-5)


def test_ccmetrics_matches_ccdice_random_batch():
    spatial = (16, 16, 16)
    seeds = list(range(8))

    preds = []
    gts = []
    idxs = []
    cc_dice = CCDiceMetric(cc_reduction="patient")

    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        gt_fg = torch.randint(0, 2, (1, *spatial), generator=generator, dtype=torch.float32)
        gt = torch.stack([1.0 - gt_fg, gt_fg], dim=1)

        pred_fg = torch.randint(0, 2, (1, *spatial), generator=generator, dtype=torch.float32)
        pred = torch.stack([1.0 - pred_fg, pred_fg], dim=1)

        preds.append(pred.numpy())
        gts.append(gt.numpy())

        cc_dice(y_pred=pred.numpy(), y=gt.numpy())

        idxs.append(torch.argmax(gt, dim=1, keepdim=True).to(torch.int64))

    baseline = float((1.0 - cc_dice.cc_aggregate()).mean())

    batch_pred = torch.from_numpy(np.concatenate(preds, axis=0))
    batch_idx = torch.from_numpy(torch.cat(idxs, dim=0).numpy())

    module = TorchCCMetrics(metric=dice_fn, activation=None)
    torch_score = module(batch_pred, batch_idx).item()

    assert torch_score == pytest.approx(baseline, rel=0.01, abs=1e-2)
