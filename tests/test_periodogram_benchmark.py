import time

import numpy as np
import pytest

from feature_extractor.periodogram import PeriodogramFeatureExtractor
from feature_extractor.utils import to_numpy


@pytest.fixture(scope="module")
def extractor():
    return PeriodogramFeatureExtractor()


@pytest.fixture(scope="module")
def sampling_rate():
    return 4096.0


def _random_epoch(
    rng: np.random.Generator,
    n_epochs: int,
    n_frames: int,
    n_channels: int,
    n_samples: int,
) -> np.ndarray:
    return rng.standard_normal(
        (n_epochs, n_frames, n_channels, n_samples), dtype=np.float32
    )


def test_periodogram_features_shape(extractor, sampling_rate):
    rng = np.random.default_rng(123)
    n_epochs, n_frames, n_channels, n_samples = 12, 96, 2, 4096
    X = _random_epoch(rng, n_epochs, n_frames, n_channels, n_samples)

    feats = to_numpy(extractor.get_feats(X, sampling_rate))

    n_bands = len(extractor.freq_bands)
    expected_shape = (n_epochs, n_frames, n_channels, n_bands * 2)
    assert feats.shape == expected_shape
    assert np.isfinite(feats).all()
    assert len(extractor.feat_names(n_channels)) == n_channels * n_bands * 2


@pytest.mark.parametrize("normalize", [False, True])
def test_periodogram_normalization_branch(normalize, sampling_rate):
    rng = np.random.default_rng(7)
    extractor = PeriodogramFeatureExtractor(normalize=normalize)
    X = _random_epoch(rng, 4, 8, 4, 1024)

    feats = to_numpy(extractor.get_feats(X, sampling_rate))
    assert np.isfinite(feats).all()


def test_periodogram_handles_single_sample(sampling_rate):
    rng = np.random.default_rng(11)
    extractor = PeriodogramFeatureExtractor(
        freq_bands=[(0.0, sampling_rate / 4)], normalize=False
    )
    X = _random_epoch(rng, 1, 1, 1, 1)

    feats = to_numpy(extractor.get_feats(X, sampling_rate))

    assert feats.shape == (1, 1, 1, 2)
    assert np.isfinite(feats).all()


@pytest.mark.performance
@pytest.mark.parametrize(
    ("n_epochs", "n_frames", "n_channels", "n_samples"),
    [
        (12, 96, 2, 4096),
        (12, 96, 16, 4096),
    ],
)
def test_periodogram_benchmark(
    extractor,
    sampling_rate,
    n_epochs,
    n_frames,
    n_channels,
    n_samples,
    record_property,
):
    rng = np.random.default_rng(321)
    X = _random_epoch(rng, n_epochs, n_frames, n_channels, n_samples)

    to_numpy(
        extractor.get_feats(X, sampling_rate)
    )  # warmup to exclude JIT compile time

    start = time.perf_counter()
    feats = to_numpy(extractor.get_feats(X, sampling_rate))
    elapsed = time.perf_counter() - start

    record_property(
        "periodogram_runtime_seconds",
        {
            "shape": [n_epochs, n_frames, n_channels, n_samples],
            "elapsed": elapsed,
        },
    )

    print(
        f"[periodogram-benchmark] shape=({n_epochs}, {n_frames}, {n_channels}, {n_samples}) "
        f"elapsed={elapsed:.6f}s"
    )

    n_bands = len(extractor.freq_bands)
    assert feats.shape == (n_epochs, n_frames, n_channels, n_bands * 2)
