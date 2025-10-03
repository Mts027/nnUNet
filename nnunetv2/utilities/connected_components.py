import torch
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt  # type: ignore
    from cupyx.scipy.ndimage import label as cp_label  # type: ignore
    _CUPY_IMPORT_ERROR = None
    _HAS_CUPY = True
except Exception as exc:  # pragma: no cover - CuPy required at runtime
    cp = None  # type: ignore
    cp_distance_transform_edt = None  # type: ignore
    cp_label = None  # type: ignore
    _CUPY_IMPORT_ERROR = exc
    _HAS_CUPY = False


def get_cc(y: torch.Tensor, do_bg: bool):
    """Assign per-channel connected-component ids on the GPU.

    Args:
        y: Prediction or target tensor shaped ``[B, C, D, H, W]``.
        do_bg: Whether to keep channel 0 (usually background) in the assignments.
    """
    assert y.ndim == 5
    cc_assignments = []
    for batch_index in range(y.shape[0]):
        per_channel = []
        for channel_index in range(0 if do_bg else 1, y.shape[1]):
            if not _HAS_CUPY:
                raise RuntimeError("CuPy with CUDA support is required for connected components.") from _CUPY_IMPORT_ERROR
            mask = y[batch_index, channel_index] > 0
            cc_per_channel = torch.from_dlpack(
                connected_components_cupy(mask, dlpack=True)
            ).to(device=y.device, dtype=torch.int64)
            per_channel.append(cc_per_channel)
        per_channel = torch.stack(per_channel, dim=0)
        cc_assignments.append(per_channel)

    cc_assignments = torch.stack(cc_assignments, dim=0)
    cc_assignments = cc_assignments.to(device=y.device)

    if do_bg:
        assert cc_assignments.shape == y.shape
    else:
        y_s = list(y.shape)
        y_s[1] -= 1

        assert list(cc_assignments.shape) == y_s

    assert cc_assignments.dtype == torch.int64
    return cc_assignments


def get_voronoi(
    y: torch.Tensor, do_bg: bool,
) -> torch.Tensor:
    """Compute per-channel Voronoi maps around foreground components on the GPU.

    Args:
        y: One-hot tensor with foreground channels, shaped ``[B, C, D, H, W]``.
        do_bg: Whether to keep the background channel in the output structure.
    """
    assert y.ndim == 5

    cc_assignments = []
    for batch_index in range(y.shape[0]):
        per_channel = []
        for channel_index in range(0 if do_bg else 1, y.shape[1]):
            if not _HAS_CUPY:
                raise RuntimeError("CuPy with CUDA support is required for Voronoi computation.") from _CUPY_IMPORT_ERROR
            mask = y[batch_index, channel_index] > 0
            per_channel.append(
                torch.from_dlpack(
                    compute_voronoi_cupy(mask, dlpack=True)
                ).to(device=y.device, dtype=torch.int64)
            )
        per_channel = torch.stack(per_channel, dim=0)
        cc_assignments.append(per_channel)

    cc_assignments = torch.stack(cc_assignments, dim=0)
    cc_assignments = cc_assignments.to(device=y.device)

    if do_bg:
        assert cc_assignments.shape == y.shape
    else:
        y_s = list(y.shape)
        y_s[1] -= 1

        assert list(cc_assignments.shape) == y_s

    assert cc_assignments.dtype == torch.int64
    return cc_assignments

def compute_voronoi_cupy(labels, dlpack: bool = False):
    """Generate GPU Voronoi partitions for a binary mask via CuPy distance transforms.

    Args:
        labels: Boolean-like array on CPU or GPU representing the foreground mask.
        dlpack: If ``True`` treat ``labels`` as a DLPack capsule for zero-copy transfer.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import label as cp_label
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt

    # 1) Move input mask to GPU
    if dlpack:
        cc_gpu_in = cp.from_dlpack(labels)
    else:
        cc_gpu_in = cp.asarray(labels)
    mask = cc_gpu_in > 0

    # 2) Build a 3×3×3 all‑ones structuring element → 26‑connectivity
    #    Non‑zero entries in `structure` are considered neighbors.
    structure = cp.ones((3, 3, 3), dtype=bool)

    # 3) GPU connected‑component labeling with full (26‑way) connectivity
    labeled_cc, _ = cp_label(mask, structure=structure)

    # 4) Compute GPU EDT (indices only) on the inverted mask
    inv_mask = labeled_cc == 0
    indices = cp_distance_transform_edt(
        inv_mask, return_distances=False, return_indices=True, float64_distances=True
    )

    # 5) Assign each voxel the label of its nearest component
    vor_gpu = labeled_cc[tuple(indices)]

    # 6) Bring result back to host
    if dlpack:
        return vor_gpu
    else:
        return cp.asnumpy(vor_gpu)


def connected_components_cupy(labels, dlpack: bool = False):
    """Label connected components on GPU using CuPy's ndimage utilities.

    Args:
        labels: Boolean-like array containing the mask to be labeled.
        dlpack: If ``True`` interpret ``labels`` as a DLPack capsule.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import label as cp_label
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt

    # 1) Move input mask to GPU
    if dlpack:
        cc_gpu_in = cp.from_dlpack(labels)
    else:
        cc_gpu_in = cp.asarray(labels)
    mask = cc_gpu_in > 0

    # 2) Build a 3×3×3 all‑ones structuring element → 26‑connectivity
    #    Non‑zero entries in `structure` are considered neighbors.
    structure = cp.ones((3, 3, 3), dtype=bool)

    # 3) GPU connected‑component labeling with full (26‑way) connectivity
    labeled_cc, _ = cp_label(mask, structure=structure)

    if dlpack:
        return labeled_cc
    else:
        return cp.asnumpy(labeled_cc)
