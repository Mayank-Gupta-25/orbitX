"""OrbitX: reinforcement learning for autonomous satellite collision avoidance."""

import io
import torch

# Fix for PyTorch 2.4+ on Windows: reading zip streams directly from ZipExtFile
# triggers internal miniz C++ read errors. Buffering into BytesIO enables full seeking.
_orig_torch_load = torch.load


def _safe_torch_load(f, *args, **kwargs):
    if hasattr(f, "read") and not isinstance(f, (io.BytesIO, io.BufferedReader)):
        try:
            content = f.read()
            f = io.BytesIO(content)
        except Exception:
            pass
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _safe_torch_load

from .env import OrbitEnv  # noqa: E402

__version__ = "0.1.0"
__all__ = ["OrbitEnv"]
