"""Decomposing a Target spectrum onto a Palette.

Model (RH2009):  T(f_i) ~ sum_j X_ij * alpha_j,  X = Palette.matrix.

Methods
  dot       projection of T on each sound alone (RH2009 "DOT")
  pinv_fit  unconstrained least squares via the pseudo-inverse (RH2009
            "SVD"); can return negative weights
  nnls_fit  least squares with alpha_j >= 0 (Lawson-Hanson, scipy),
            optionally weighted by 1/sigma_i as in Numerical Recipes 15.4

Playability (one note per player, RH2009 "CHI") is Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import nnls

from .bank import Palette


def chi2(target: np.ndarray, model: np.ndarray,
         sigma: Optional[np.ndarray] = None) -> float:
    """Sum of squared (optionally sigma-weighted) residuals."""
    r = np.asarray(target, float) - np.asarray(model, float)
    if sigma is not None:
        r = r / sigma
    return float(r @ r)


@dataclass
class Fit:
    """Result of a decomposition. ``weights`` follow ``names`` order."""
    names: list
    weights: np.ndarray
    target: np.ndarray
    matrix: np.ndarray = field(repr=False)
    method: str = ""
    sigma: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def model(self) -> np.ndarray:
        """The fitted spectrum, X @ alpha."""
        return self.matrix @ self.weights

    @property
    def chi2(self) -> float:
        return chi2(self.target, self.model, self.sigma)

    @property
    def relative_error(self) -> float:
        """||T - X alpha|| / ||T|| (unweighted)."""
        return float(np.linalg.norm(self.target - self.model) / np.linalg.norm(self.target))

    @property
    def order(self) -> np.ndarray:
        """Indices sorted by weight, largest first (the notebooks' ``wind``)."""
        return np.argsort(-self.weights, kind="stable")

    def ranked(self, n: Optional[int] = None, nonzero: bool = True,
               rtol: float = 0.0) -> list:
        """[(name, weight), ...] largest first.

        With ``nonzero`` only weights above ``rtol * max(weights)`` are kept
        (use a small rtol such as 1e-9 to drop round-off).
        """
        out = [(self.names[i], float(self.weights[i])) for i in self.order]
        if nonzero:
            cut = rtol * max(float(self.weights.max()), 0.0)
            out = [p for p in out if p[1] > cut]
        return out[:n] if n is not None else out

    def top(self, n: int, refit: bool = False) -> "Fit":
        """Keep the n largest weights (notebook 1.5 / toolkeepN.py).

        With ``refit=False`` the kept weights are left as they were, which is
        what the 2020 mock-up did. With ``refit=True`` NNLS is re-run on the
        kept sounds only, which usually lowers chi2.
        """
        keep = self.order[:n]
        sub = Palette([self.names[i] for i in keep], self.matrix[:, keep])
        if refit:
            return nnls_fit(sub, self.target, sigma=self.sigma)
        return Fit(sub.names, self.weights[keep].copy(), self.target, sub.matrix,
                   self.method + f"/top{n}", self.sigma)

    def kept_fraction(self, n: int) -> float:
        """Share of the total weight carried by the top n sounds."""
        w = self.weights[self.order]
        return float(w[:n].sum() / w.sum()) if w.sum() > 0 else 0.0

    def summary(self, n: int = 10) -> str:
        lines = [f"{self.method}: {len(self.names)} sounds, "
                 f"relative error {self.relative_error:.3f}"]
        lines += [f"  {w:8.4f}  {name}" for name, w in self.ranked(n)]
        return "\n".join(lines)


def _weighted(palette: Palette, target: np.ndarray, sigma):
    X = np.asarray(palette.matrix, float)
    T = np.asarray(target, float)
    if T.shape[0] != X.shape[0]:
        raise ValueError(f"target has {T.shape[0]} bins, palette {X.shape[0]}")
    if sigma is None:
        return X, T
    sigma = np.asarray(sigma, float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")
    return X / sigma[:, None], T / sigma


def dot(palette: Palette, target: np.ndarray, normalize: bool = False) -> Fit:
    """Projection p_j = sum_i T_i X_ij of the Target on each sound alone.

    With ``normalize=True`` each column is scaled to unit length first, so
    the score measures shape (cosine similarity) rather than loudness.
    The returned Fit's weights are the projections; its ``model`` is not a
    least-squares fit and should not be compared with NNLS chi2.
    """
    X, T = _weighted(palette, target, None)
    if normalize:
        X = X / np.linalg.norm(X, axis=0, keepdims=True)
        T = T / np.linalg.norm(T)
    return Fit(list(palette.names), X.T @ T, np.asarray(target, float),
               palette.matrix, "dot" + ("-cos" if normalize else ""))


def pinv_fit(palette: Palette, target: np.ndarray,
             sigma: Optional[np.ndarray] = None) -> Fit:
    """Unconstrained least squares (RH2009 'SVD'); weights may be negative."""
    A, b = _weighted(palette, target, sigma)
    w = np.linalg.pinv(A) @ b
    return Fit(list(palette.names), w, np.asarray(target, float),
               palette.matrix, "pinv", sigma)


def nnls_fit(palette: Palette, target: np.ndarray,
             sigma: Optional[np.ndarray] = None,
             maxiter: Optional[int] = None) -> Fit:
    """Non-negative least squares: min ||(X alpha - T)/sigma|| s.t. alpha >= 0.

    ``sigma=None`` means every bin counts equally, which is what the 2020
    code did. Choosing sigma (relative, perceptual, ...) is a Phase 2 task.
    """
    A, b = _weighted(palette, target, sigma)
    kw = {} if maxiter is None else {"maxiter": maxiter}
    w, _ = nnls(A, b, **kw)
    return Fit(list(palette.names), w, np.asarray(target, float),
               palette.matrix, "nnls", sigma)
