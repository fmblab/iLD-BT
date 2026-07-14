"""Colormap for NW terrain and sequence-landscape figures: white -> viridis(0.55) -> yellow, Normalize(0, 3.8).

Figures: NW terrain / sequence-landscape panels (Fig 3, Fig 5, Fig 7; Fig S13, S16).
"""

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np

_viridis = mcm.get_cmap('viridis')
_VSTART  = 0.55
_WT_FRAC = 1.0 / 3.8
_n_below = max(1, round(_WT_FRAC * 256))
_n_above = 256 - _n_below
_white   = np.array([1.0, 1.0, 1.0, 1.0])
_v_start = np.array(_viridis(_VSTART))
_ramp    = np.array([_white * (1 - t) + _v_start * t for t in np.linspace(0, 1, _n_below)])
_tail    = _viridis(np.linspace(_VSTART, 1.0, _n_above))
cmap     = mcolors.LinearSegmentedColormap.from_list('landscape', np.vstack([_ramp, _tail]), N=256)
norm     = mcolors.Normalize(vmin=0.0, vmax=3.8)

SCATTER_KW  = dict(edgecolors='black', linewidths=1.0)
WT_CONTOUR  = dict(colors='white', linewidths=1.5, linestyles='--')
CB_TICKS    = [0, 0.5, 1.0, 2.0, 3.0, 3.8]
