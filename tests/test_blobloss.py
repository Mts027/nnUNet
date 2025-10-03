import numpy as np
import pytest
import torch
from scipy import ndimage

pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for BlobLoss tests",
)

from tests.blob_loss import compute_blob_loss_multi
from nnunetv2.training.loss.instance_losses import BlobLoss as TorchBlobLoss


def blob_loss_metric(y_pred: torch.Tensor, y: torch.Tensor, pred_mask, true_mask) -> torch.Tensor:
    eps = 1e-7
    mask = pred_mask.to(dtype=y_pred.dtype)
    pred_fg = torch.clamp(y_pred[1] * mask, min=eps, max=1.0 - eps)
    target_fg = y[1] * mask  # y already one-hot, mask keeps background + blob voxels
    return torch.nn.functional.binary_cross_entropy(pred_fg, target_fg, reduction="mean")


_CC_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def _baseline_blob_loss(y_pred_np: np.ndarray, y_np: np.ndarray) -> float:
    batch = y_np.shape[0]
    scores: list[float] = []
    for batch_index in range(batch):
        outputs = torch.from_numpy(y_pred_np[batch_index : batch_index + 1, 1:2]).clamp_(1e-7, 1.0 - 1e-7)

        # Blob loss expects instance labels with unique ids per component; derive them here.
        foreground = np.argmax(y_np[batch_index], axis=0).astype(np.uint8)
        connected, _ = ndimage.label(foreground, structure=_CC_STRUCTURE)
        labels = torch.from_numpy(connected[np.newaxis, np.newaxis, ...].astype(np.int64))

        loss = compute_blob_loss_multi(
            criterion=torch.nn.functional.binary_cross_entropy,
            network_outputs=outputs,
            multi_label=labels,
        )
        scores.append(float(loss))
    return float(np.mean(scores)) if scores else 0.0


@pytest.mark.parametrize("spatial", [(24, 24, 24)])
def test_blobloss_matches_reference_empty(spatial):
    batch = 1
    y_pred_np = np.zeros((batch, 2, *spatial), dtype=np.float32)
    y_pred_np[:, 0] = 1.0
    y_np = y_pred_np.copy()

    baseline = _baseline_blob_loss(y_pred_np, y_np)

    device = torch.device("cuda")
    module = TorchBlobLoss(metric=blob_loss_metric, activation=None).to(device)
    y_pred_t = torch.from_numpy(y_pred_np).to(device)
    y_idx = torch.zeros((batch, 1, *spatial), dtype=torch.int64, device=device)

    torch_score = module(y_pred_t, y_idx).detach().cpu().item()

    assert torch_score == pytest.approx(baseline, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("spatial", [(24, 24, 24)])
def test_blobloss_matches_reference_gt_nonempty_pred_empty(spatial):
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

    baseline = _baseline_blob_loss(y_pred_np, y_np)

    device = torch.device("cuda")
    module = TorchBlobLoss(metric=blob_loss_metric, activation=None).to(device)
    y_pred_t = torch.from_numpy(y_pred_np).to(device)
    y_idx = torch.from_numpy(np.argmax(y_np, axis=1, keepdims=True).astype(np.int64)).to(device)

    torch_score = module(y_pred_t, y_idx).detach().cpu().item()

    assert torch_score == pytest.approx(baseline, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "spatial,p_fg_gt,p_fg_pred,seed",
    [
        ((16, 16, 16), 0.5, 0.5, 0),
        ((12, 18, 20), 0.2, 0.7, 1),
        ((20, 20, 12), 0.8, 0.3, 2),
    ],
)
def test_blobloss_matches_reference_random_batch(spatial, p_fg_gt, p_fg_pred, seed):
    batch = 4
    rng = np.random.default_rng(seed)

    gt_fg = (rng.random((batch, *spatial), dtype=np.float32) < p_fg_gt).astype(np.float32)
    pred_fg = rng.random((batch, *spatial), dtype=np.float32)
    pred_fg = np.clip(pred_fg, 1e-5, 1 - 1e-5)

    gt_np = np.stack([1.0 - gt_fg, gt_fg], axis=1)
    pred_np = np.stack([1.0 - pred_fg, pred_fg], axis=1)

    baseline = _baseline_blob_loss(pred_np, gt_np)

    device = torch.device("cuda")
    batch_pred = torch.from_numpy(pred_np).to(device)
    batch_idx = torch.from_numpy(np.argmax(gt_np, axis=1, keepdims=True).astype(np.int64)).to(device)

    module = TorchBlobLoss(metric=blob_loss_metric, activation=None).to(device)
    torch_score = module(batch_pred, batch_idx).detach().cpu().item()

    assert torch_score == pytest.approx(baseline, rel=1e-5, abs=1e-5)
