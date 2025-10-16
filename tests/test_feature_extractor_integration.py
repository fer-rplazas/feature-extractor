import numpy as np
import pytest
import time

from feature_extractor import FeatureExtractor


def _synthetic_signal(
    n_epochs: int,
    n_channels: int,
    n_samples: int,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    # Mixture of sinusoids with noise to mimic neural-like structure.
    time = np.linspace(0, 1, n_samples, endpoint=False)
    freqs = np.array([8.0, 12.0, 30.0])
    signal = (
        np.sin(2 * np.pi * freqs[0] * time)
        + 0.5 * np.sin(2 * np.pi * freqs[1] * time)
        + 0.25 * np.sin(2 * np.pi * freqs[2] * time)
    )

    noise = rng.standard_normal((n_epochs, n_channels, n_samples)) * 0.05
    base = signal[None, None, :]
    return (base + noise).astype(np.float32)


@pytest.mark.parametrize(
    ("fs", "n_channels", "n_jobs"),
    [
        (4096.0, 1, 96),
        (4096.0, 16, 96),
    ],
)
def test_feature_extractor_default_pipeline(fs, n_channels, n_jobs):
    n_epochs, n_channels, n_samples = 32, n_channels, int(fs) * 8

    time_series = _synthetic_signal(n_epochs, n_channels, n_samples)

    extractor = FeatureExtractor(fs=fs, n_jobs=n_jobs, device="cpu")
    start = time.perf_counter()
    extractor.calculate_features(time_series)
    extractor.causal_bl_correct(len_time=5.0)
    duration = time.perf_counter() - start
    print(
        f"Shape: {time_series.shape} – Feature extraction took {duration:.3f} seconds."
    )
    feats_before = extractor.features.copy(deep=True)
    print(feats_before)

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
    extractor.causal_bl_correct(len_time=5.0)
    np.testing.assert_allclose(
        feats_before.values,
        extractor.features.values,
        rtol=1e-5,
        atol=1e-6,
    )


if __name__ == "__main__":
    test_feature_extractor_default_pipeline(4096.0, 16, 96)
