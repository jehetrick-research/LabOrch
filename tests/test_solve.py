import numpy as np
import pytest

from laborch import Palette, nnls_fit, pinv_fit, dot
from laborch.audio import weights_to_db


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.random((200, 6))
    T = 0.5 * X[:, 0] + 0.2 * X[:, 3]
    return Palette([f"s{i}" for i in range(6)], X), T


def test_nnls_recovers_exact_mixture(toy):
    pal, T = toy
    f = nnls_fit(pal, T)
    assert np.allclose(f.weights, [0.5, 0, 0, 0.2, 0, 0], atol=1e-8)
    assert f.relative_error < 1e-8
    assert [n for n, _ in f.ranked(rtol=1e-9)] == ["s0", "s3"]


def test_sigma_weighting_leaves_exact_solution(toy):
    pal, T = toy
    f = nnls_fit(pal, T, sigma=np.linspace(1, 3, 200))
    assert np.allclose(f.weights, [0.5, 0, 0, 0.2, 0, 0], atol=1e-8)


def test_top_and_refit(toy):
    pal, T = toy
    f = nnls_fit(pal, T + 0.01)
    t = f.top(2)
    assert len(t.names) == 2 and np.allclose(t.weights, f.weights[f.order[:2]])
    r = f.top(2, refit=True)
    assert r.chi2 <= t.chi2 + 1e-12


def test_pinv_and_dot_shapes(toy):
    pal, T = toy
    assert pinv_fit(pal, T).weights.shape == (6,)
    d = dot(pal, T, normalize=True)
    assert np.all(d.weights <= 1 + 1e-12)


def test_weights_to_db():
    db = weights_to_db([1.0, 0.1, 0.0])
    assert np.allclose(db, [0, -20, -60])


def test_weights_to_db_power():
    assert np.allclose(weights_to_db([1.0, 0.1], kind="power"), [0, -10])
    assert np.allclose(weights_to_db([2.0, 0.5], kind="power", relative=False),
                       [10 * np.log10(2), 10 * np.log10(0.5)])


def test_nnls_is_scale_invariant(toy):
    """Tiny power values (Bank v2) give the same weights."""
    pal, T = toy
    small = Palette(pal.names, pal.matrix * 1e-9)
    f = nnls_fit(small, T * 1e-9)
    assert np.allclose(f.weights, [0.5, 0, 0, 0.2, 0, 0], atol=1e-8)
