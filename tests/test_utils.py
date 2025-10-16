import time

import numpy as np
import pytest

from feature_extractor.utils import causal_bl_correct, create_trailing_frames, to_numpy


def _reference_frames(
    time_series: np.ndarray,
    frame_length: int,
    hop_length: int,
    initial_offset: int,
) -> np.ndarray:
    n_trials, n_channels, _ = time_series.shape
    num_frames = (time_series.shape[-1] - initial_offset) // hop_length + 1
    expected = np.empty(
        (n_trials, num_frames, n_channels, frame_length), dtype=time_series.dtype
    )

    for trial in range(n_trials):
        for frame_idx in range(num_frames):
            start = initial_offset + frame_idx * hop_length
            expected[trial, frame_idx] = time_series[
                trial, :, start - frame_length : start
            ]

    return expected


@pytest.mark.parametrize(
    ("frame_length", "hop_length", "initial_offset"),
    [
        (4, 2, None),
        (5, 3, 8),
    ],
)
def test_create_trailing_frames_matches_reference(
    frame_length,
    hop_length,
    initial_offset,
):
    rng = np.random.default_rng(123)
    data = rng.integers(0, 100, size=(2, 3, 20), dtype=np.int32)

    frames = to_numpy(create_trailing_frames(data, frame_length, hop_length, initial_offset))

    offset = initial_offset if initial_offset is not None else frame_length
    expected = _reference_frames(data, frame_length, hop_length, offset)

    assert frames.dtype == data.dtype
    np.testing.assert_array_equal(frames, expected)


def test_create_trailing_frames_invalid_shape():
    with pytest.raises(AssertionError, match="Time series should be a 3D array"):
        to_numpy(create_trailing_frames(np.zeros((2, 10)), frame_length=4, hop_length=2))


def test_create_trailing_frames_preserves_dtype():
    data = np.linspace(0.0, 1.0, 40, dtype=np.float32).reshape(2, 2, 10)

    frames = to_numpy(create_trailing_frames(data, frame_length=5, hop_length=3))

    assert frames.dtype == np.float32


def test_causal_bl_correct_constant_signal():
    data = np.ones((1, 20, 3, 2), dtype=np.float32)

    corrected = causal_bl_correct(data)

    assert corrected.shape == data.shape
    assert np.allclose(corrected, 0.0)


def test_causal_bl_correct_returns_stable_scale():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1, 200, 2, 3)).astype(np.float32)

    corrected = causal_bl_correct(data, len_scale=50, warmup_points=0)

    # After normalization, expect approximately unit variance per feature.
    variances = corrected.var(axis=1, dtype=np.float64)
    assert np.all(variances > 0.1)
    assert np.all(variances < 5.0)


@pytest.mark.performance
@pytest.mark.parametrize(
    ("n_epochs", "n_frames", "n_win_len", "n_features"),
    [
        (32, 96, 2, 32),
        (32, 96, 2, 320),
    ],
)
def test_causal_bl_correct_performance(
    n_epochs,
    n_frames,
    n_win_len,
    n_features,
    record_property,
):
    rng = np.random.default_rng(2024)
    data = rng.standard_normal(
        (n_epochs, n_frames, n_win_len, n_features), dtype=np.float32
    )

    causal_bl_correct(data.copy())  # warmup compile run

    start = time.perf_counter()
    result = causal_bl_correct(data.copy())
    elapsed = time.perf_counter() - start

    record_property(
        "causal_bl_correct_runtime_seconds",
        {
            "shape": [n_epochs, n_frames, n_win_len, n_features],
            "elapsed": elapsed,
        },
    )
    print(
        f"[causal-bl-benchmark] shape=({n_epochs}, {n_frames}, {n_win_len}, {n_features}) "
        f"elapsed={elapsed:.6f}s"
    )

    assert np.isfinite(result).all()
