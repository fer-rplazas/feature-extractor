import time

import numpy as np
import pytest

from feature_extractor.periodogram import PeriodogramFeatureExtractor


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

    feats = extractor.get_feats(X, sampling_rate)

    n_bands = len(extractor.freq_bands)
    expected_shape = (n_epochs, n_frames, n_channels, n_bands * 2)
    assert feats.shape == expected_shape


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

    extractor.get_feats(X, sampling_rate)  # warmup to exclude numba jit compile time

    start = time.perf_counter()
    feats = extractor.get_feats(X, sampling_rate)
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
