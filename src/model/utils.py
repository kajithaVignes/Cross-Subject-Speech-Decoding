import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import re

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, step, extra = None, folder="checkpoints", base_name="ckpt"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    is_manual = base_name.startswith("_")

    if is_manual:
        filename = f"{base_name}.pt"
    else:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"{base_name}_{timestamp}.pt"

    full_path = folder / filename

    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer else None,
        "step": step,
        "extra": extra or {},
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(ckpt, full_path)
    print(f"Saved checkpoint to {full_path}")

    if not is_manual:
        auto_ckpts = sorted([f for f in folder.glob(f"{base_name}_*.pt") if not f.name.startswith("_")],
                            key=lambda x: x.stat().st_mtime)
        for old_ckpt in auto_ckpts[:-5]:
            old_ckpt.unlink()
            print(f"Deleted old checkpoint: {old_ckpt.name}")



_SESSION_WEIGHT_RE = re.compile(r"^linear\.transforms\.([^\.]+)\.weight$")

def _infer_transforms_dims_and_sessions(state_dict):
    sessions = set()
    in_dim = out_dim = None

    for k, v in state_dict.items():
        m = _SESSION_WEIGHT_RE.match(k)
        if m:
            sessions.add(m.group(1))
            if in_dim is None:
                out_dim, in_dim = v.shape[0], v.shape[1]

    if not sessions:
        return set(), None, None
    return sessions, in_dim, out_dim


def ensure_transforms_exist(model, state_dict, device):
    """
    Ensures model.linear.transforms has all session-specific Linear layers
    present in the checkpoint state_dict.
    """
    sessions, in_dim, out_dim = _infer_transforms_dims_and_sessions(state_dict)

    if not sessions:
        return

    if not hasattr(model, "linear") or not hasattr(model.linear, "transforms"):
        raise AttributeError(
            "Checkpoint contains linear.transforms.<session>.* but model has no model.linear.transforms. "
            "Check your Encoder definition (path/name mismatch)."
        )

    if not isinstance(model.linear.transforms, (nn.ModuleDict,)):
        raise TypeError("model.linear.transforms must be an nn.ModuleDict to register per-session modules.")

    for s in sorted(sessions):
        if s not in model.linear.transforms:
            model.linear.transforms[s] = nn.Linear(in_dim, out_dim).to(device)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"]

    ensure_transforms_exist(model, state, device=device)

    model.load_state_dict(state, strict=True)

    if optimizer is not None and ckpt.get("optim_state") is not None:
        optimizer.load_state_dict(ckpt["optim_state"])

    step = int(ckpt.get("step", 0))
    extra = ckpt.get("extra", {})
    print(f"✅ loaded checkpoint from {path} (step={step})")
    return step, extra
