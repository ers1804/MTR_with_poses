"""AVO search hook: load a candidate pose-integration overlay from POSE_OVERLAY_DIR.

The AVO harness (avo_harness/) evolves a single file, ``overlay.py``, that controls how
body pose enters the model. Everything else in this repository is frozen for the search.

Overlay contract (all attributes required unless noted):

    FEATURE_DIM : int
        Width F of the per-timestep pose feature vector the overlay produces.
    MODE : str, "early" | "module"
        "early"  — features are concatenated onto the agent's per-timestep input
                   channels in the dataset; the encoder is untouched
                   (NUM_INPUT_ATTR_AGENT must be 29 + FEATURE_DIM at train time).
        "module" — features are shipped to the encoder as ``obj_poses`` (C,N,T,F) and
                   fused by the overlay's own module (USE_POSE_ENCODER +
                   USE_OVERLAY_FUSION must be set; POSE_INPUT_DIM = FEATURE_DIM).
    derive_features(pose_6d, headings) -> np.ndarray
        pose_6d  (..., T, 144) 6D rotations (root 0:6, 23 body joints after);
        headings (..., T) agent-centric travel heading of the same agent.
        Returns (..., T, FEATURE_DIM) float32. Invalid frames are zeroed by the
        caller afterwards; the function itself must return finite values everywhere.
    build_fusion(d_model, feature_dim) -> torch.nn.Module          [MODE="module" only]
        Module with forward(agent_feat (B,N,D), pose_feats (B,N,T,F),
        pose_mask (B,N,T) bool) -> (B,N,D). Must be zero-safe for agents whose
        mask is all-False.

The loader is intentionally dumb: no reload, no fallback chain. If POSE_OVERLAY_DIR is
unset, the repository's built-in behaviour is used and this module returns None.
"""

import importlib.util
import os
import sys

_cache = None
_loaded_from = None


def get_overlay():
    """Return the overlay module, or None when POSE_OVERLAY_DIR is unset."""
    global _cache, _loaded_from
    d = os.environ.get('POSE_OVERLAY_DIR')
    if not d:
        return None
    if _cache is not None and _loaded_from == d:
        return _cache
    path = os.path.join(d, 'overlay.py')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'POSE_OVERLAY_DIR set but no overlay.py in {d}')
    spec = importlib.util.spec_from_file_location('pose_overlay_candidate', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pose_overlay_candidate'] = mod
    spec.loader.exec_module(mod)
    for attr in ('FEATURE_DIM', 'MODE', 'derive_features'):
        if not hasattr(mod, attr):
            raise AttributeError(f'overlay.py missing required attribute {attr!r}')
    if mod.MODE not in ('early', 'module'):
        raise ValueError(f'overlay MODE must be "early" or "module", got {mod.MODE!r}')
    if mod.MODE == 'module' and not hasattr(mod, 'build_fusion'):
        raise AttributeError('overlay MODE="module" requires build_fusion()')
    _cache, _loaded_from = mod, d
    return mod
