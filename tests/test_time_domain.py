import numpy as np
import pytest

from feature_extractor.time_domain import TimeDomainFeatureExtractor, hjorth
from feature_extractor.utils import to_numpy


def _manual_time_domain_features(X: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(X, axis=-1, keepdims=True)
    energy = np.mean(np.square(X), axis=-1, keepdims=True)

    centered = X - mean
    variance = np.maximum(np.var(X, axis=-1), 0.0)
    variance_safe = variance + eps

    skew_numer = np.mean(centered**3, axis=-1)
    kurt_numer = np.mean(centered**4, axis=-1)

    skew = np.where(
        variance > 0, skew_numer / np.power(variance_safe, 1.5), 0.0
    )[..., None]
    kurtosis = np.where(
        variance > 0, kurt_numer / np.square(variance_safe) - 3.0, 0.0
    )[..., None]

    if X.shape[-1] > 1:
        first_deriv = np.diff(X, axis=-1)
    else:
        first_deriv = np.zeros(X.shape[:-1] + (0,), dtype=X.dtype)

    if first_deriv.shape[-1] > 1:
        second_deriv = np.diff(first_deriv, axis=-1)
    else:
        second_deriv = np.zeros(first_deriv.shape[:-1] + (0,), dtype=X.dtype)

    if first_deriv.shape[-1] > 0:
        var_d1 = np.maximum(np.var(first_deriv, axis=-1), 0.0)
    else:
        var_d1 = np.zeros(X.shape[:-1], dtype=X.dtype)

    if second_deriv.shape[-1] > 0:
        var_d2 = np.maximum(np.var(second_deriv, axis=-1), 0.0)
    else:
        var_d2 = np.zeros(X.shape[:-1], dtype=X.dtype)

    activity = variance
    mobility = np.sqrt(var_d1 / (variance + eps))
    complexity = np.sqrt(var_d2 / (var_d1 + eps)) / (mobility + eps)

    hjorth = np.stack((activity, mobility, complexity), axis=-1)

    return np.concatenate([mean, energy, skew, kurtosis, hjorth], axis=-1)


def test_time_domain_features_matches_manual():
    extractor = TimeDomainFeatureExtractor()
    eps = extractor.eps

    X = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    feats = to_numpy(extractor.get_feats(X))

    expected = _manual_time_domain_features(X, eps)
    np.testing.assert_allclose(feats, expected, rtol=1e-5, atol=1e-6)


def test_time_domain_handles_constant_signal():
    extractor = TimeDomainFeatureExtractor()
    X = np.ones((2, 3, 4, 5), dtype=np.float32)

    feats = to_numpy(extractor.get_feats(X))

    assert np.isfinite(feats).all()
    means = feats[..., 0]
    energies = feats[..., 1]
    assert np.allclose(means, 1.0)
    assert np.allclose(energies, 1.0)
    assert np.allclose(feats[..., 2], 0.0)  # skew
    assert np.allclose(feats[..., 3], 0.0)  # kurt


def test_time_domain_accepts_integer_input():
    extractor = TimeDomainFeatureExtractor()
    X = np.arange(24, dtype=np.int32).reshape(1, 2, 3, 4)

    feats = to_numpy(extractor.get_feats(X))

    assert feats.dtype == np.float32
    assert feats.shape == (1, 2, 3, 7)


def test_time_domain_feature_names():
    extractor = TimeDomainFeatureExtractor()
    names = extractor.feat_names(2)

    assert len(names) == 14
    assert names[0] == "mean_ch0"
    assert names[-1] == "complex_ch1"


def test_hjorth_function_matches_extractor():
    extractor = TimeDomainFeatureExtractor()
    X = np.random.default_rng(0).standard_normal((2, 3, 4, 32)).astype(np.float32)

    hjorth_only = hjorth(X, eps=extractor.eps)
    feats = to_numpy(extractor.get_feats(X))

    np.testing.assert_allclose(hjorth_only, feats[..., -3:], rtol=1e-5, atol=1e-6)
