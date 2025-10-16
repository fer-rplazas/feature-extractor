import numpy as np
import pytest

from feature_extractor.cepstral import CepstralFeatureExtractor
from feature_extractor.utils import to_numpy


def _numpy_reference(X: np.ndarray, n_bands: int):
    X = X.astype(np.float32)
    n_samples = X.shape[-1]
    window = np.hanning(n_samples)
    spectrum = np.fft.rfft(X * window, axis=-1)
    log_mag = np.log(np.abs(spectrum) + np.finfo(np.float32).eps)
    cepstrum = np.fft.irfft(log_mag, n=n_samples, axis=-1)

    n_half = int(n_samples // 2)
    edges = [1]
    for idx in range(1, n_bands + 1):
        edge = int(n_half ** (idx / n_bands))
        edges.append(min(edge, n_half))
    edges = np.asarray(edges, dtype=int)

    feats = np.zeros(X.shape[:-1] + (n_bands * 2,), dtype=cepstrum.dtype)

    for band in range(n_bands):
        low, high = edges[band], edges[band + 1]
        if high > low:
            band_slice = cepstrum[..., low:high]
            feats[..., band] = band_slice.mean(axis=-1)
            feats[..., n_bands + band] = band_slice.std(axis=-1)
    return feats


@pytest.mark.parametrize("n_bands", [3, 10])
def test_cepstral_features_match_numpy(n_bands):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 96, 2, 4096)).astype(np.float32)

    extractor = CepstralFeatureExtractor(n_bands=n_bands)
    feats = to_numpy(extractor.get_feats(X))

    expected = _numpy_reference(X, n_bands)

    significant = np.abs(expected) > 1e-4
    np.testing.assert_allclose(
        feats[significant], expected[significant], rtol=0, atol=1e-3
    )


def test_cepstral_feat_names():
    extractor = CepstralFeatureExtractor(n_bands=3)
    names = extractor.feat_names(2)

    assert len(names) == 12
    assert names[0] == "CEP0-mean_ch0"
    assert names[-1] == "CEP2-std_ch1"


def test_cepstral_invalid_input():
    extractor = CepstralFeatureExtractor()
    with pytest.raises(ValueError, match="shape"):
        extractor.get_feats(np.zeros((2, 3, 10), dtype=np.float32))
