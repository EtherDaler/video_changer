import argparse

import torch
import torch.nn.functional as F


def load_raft(model_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load RAFT optical flow model from checkpoint."""
    try:
        from raft import RAFT
    except ImportError:
        raise ImportError(
            "RAFT not found. Clone https://github.com/princeton-vl/RAFT "
            "and add its root to PYTHONPATH."
        )

    args = argparse.Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )

    model = torch.nn.DataParallel(RAFT(args))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.module.to(device).eval()
    return model


def compute_flow_raft(
    model: torch.nn.Module,
    frame1,
    frame2,
    device: str = "cuda",
):
    """Compute dense optical flow between two RGB numpy frames using RAFT."""
    try:
        from utils.utils import InputPadder
    except ImportError:
        raise ImportError(
            "RAFT utils not found. Make sure RAFT repo root is in PYTHONPATH."
        )

    t1 = torch.from_numpy(frame1).permute(2, 0, 1).float()[None].to(device)
    t2 = torch.from_numpy(frame2).permute(2, 0, 1).float()[None].to(device)

    padder = InputPadder(t1.shape)
    t1, t2 = padder.pad(t1, t2)

    with torch.no_grad():
        _, flow = model(t1, t2, iters=20, test_mode=True)

    flow = padder.unpad(flow)
    return flow[0].permute(1, 2, 0).cpu().numpy()
