# %%
import numpy as np
import pytest
import time

from feature_extractor import FeatureExtractor
import feature_extractor

print("Feature Extractor file:", feature_extractor.__file__)

# %%


def _synthetic_signal(
    n_epochs: int,
    n_channels: int,
    n_seconds: float,
    fs: float = 4096.0,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    # Mixture of sinusoids with noise to mimic neural-like structure.
    time = np.linspace(0, n_seconds, int(n_seconds * fs))
    freqs = np.array([8.0, 12.0, 30.0])
    signal = (
        np.sin(2 * np.pi * freqs[0] * time)
        + 0.5 * np.sin(2 * np.pi * freqs[1] * time)
        + 0.25 * np.sin(2 * np.pi * freqs[2] * time)
    )

    noise = rng.standard_normal((n_epochs, n_channels, int(n_seconds * fs))) * 0.05
    base = signal[None, None, :]
    return (base + noise).astype(np.float32)


@pytest.mark.parametrize(
    ("fs", "n_channels", "n_jobs", "device"),
    [
        (4096.0, 1, 96, "cpu"),
        (4096.0, 16, 96, "cpu"),
    ],
)
def test_feature_extractor_default_pipeline(fs, n_channels, n_jobs, device):
    n_epochs, n_channels, n_seconds = 32, n_channels, 8

    from loguru import logger

    time_series = _synthetic_signal(n_epochs, n_channels, n_seconds, fs=fs)

    extractor = FeatureExtractor(fs=fs, n_jobs=n_jobs, device=device)

    start = time.perf_counter()
    extractor.calculate_features(time_series)
    extractor.causal_bl_correct(len_time=20.0)
    duration = time.perf_counter() - start

    logger.info(
        f"Shape: {time_series.shape} – Feature extraction took {duration:.3f} seconds."
    )
    feats_before = extractor.features.copy(deep=True)

    # Basic shape checks
    assert extractor.features is not None
    feats = extractor.features
    assert feats.sizes["epoch"] == n_epochs
    assert feats.sizes["win_size"] == len(extractor.win_sizes)
    assert feats.sizes["frame"] > 0
    assert feats.sizes["feature_name"] > 0

    # Ensure no NaNs/Infs and features carry expected metadata.
    assert np.isfinite(feats.values).all()
    assert "fs" in feats.attrs and feats.attrs["fs"] == fs

    # Re-run to ensure deterministic output for same input.
    extractor.calculate_features(time_series)
    extractor.causal_bl_correct(len_time=20.0)
    np.testing.assert_allclose(
        feats_before.values,
        extractor.features.values,
        rtol=1e-5,
        atol=1e-6,
    )
