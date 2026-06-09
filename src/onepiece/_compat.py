"""Compatibility helpers for supported dependency ranges."""

from __future__ import annotations

import numpy as np

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
trapezoid = getattr(np, "trapezoid", None) or np.trapz
