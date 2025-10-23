from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from jaxtyping import Float, Array

import warnings


def ensure_device_array(
    arr: Any,
    *,
    device: jax.Device | None = None,
    dtype: jnp.dtype | None = None,
) -> jax.Array:
    target_device = device

    if isinstance(arr, jax.Array):
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        if target_device is not None:
            arr = jax.device_put(arr, target_device)
        return arr

    np_value = np.asarray(arr)
    if dtype is not None:
        np_value = np_value.astype(dtype, copy=False)

    target_device = jax.devices("cpu")[0] if target_device is None else target_device

    return jnp.array(np_value, device=target_device)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, jax.Array):
        return np.asarray(jax.device_get(value))
    return np.asarray(value)


def _causal_normalize(
    sequence: jnp.ndarray,
    alpha_main_fast: jnp.ndarray,
    alpha_warmup_fast: jnp.ndarray,
    alpha_main_slow: jnp.ndarray,
    alpha_warmup_slow: jnp.ndarray,
    blend_weight: jnp.ndarray,
    min_var_fraction: jnp.ndarray,
    warmup_points: int,
    eps: float,
) -> jnp.ndarray:
    if sequence.ndim != 1:
        raise ValueError("Expected 1D sequence for causal normalization.")

    def body(
        carry: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ):
        mean_fast, mean_slow, var_num, var_den, power = carry
        value, idx = inputs

        alpha_fast = jnp.where(idx < warmup_points, alpha_warmup_fast, alpha_main_fast)
        alpha_slow = jnp.where(idx < warmup_points, alpha_warmup_slow, alpha_main_slow)
        alpha_var = alpha_fast
        alpha_power = alpha_slow

        mean_fast = mean_fast + alpha_fast * (value - mean_fast)
        mean_slow = mean_slow + alpha_slow * (value - mean_slow)
        mean_blend = blend_weight * mean_fast + (1.0 - blend_weight) * mean_slow

        var_num = (1.0 - alpha_var) * var_num + alpha_var * (value - mean_blend) ** 2
        var_den = (1.0 - alpha_var) * var_den + alpha_var

        power = (1.0 - alpha_power) * power + alpha_power * (value**2)

        var_unbiased = var_num / jnp.maximum(var_den, eps)
        min_var = min_var_fraction * power + eps
        var_clamped = jnp.maximum(var_unbiased, min_var)
        std = jnp.sqrt(var_clamped)
        normalized = (value - mean_blend) / (std + eps)

        return (mean_fast, mean_slow, var_num, var_den, power), normalized

    warmup_len = min(sequence.shape[0], max(warmup_points + 1, 2))
    initial_segment = sequence[:warmup_len]
    init_mean = jnp.mean(initial_segment, dtype=initial_segment.dtype)
    init_power = jnp.mean(initial_segment**2, dtype=initial_segment.dtype)
    centered = initial_segment - init_mean
    init_var = jnp.mean(centered**2, dtype=initial_segment.dtype)

    init_carry = (
        init_mean,
        init_mean,
        init_var,
        jnp.asarray(1.0, dtype=initial_segment.dtype),
        init_power,
    )

    _, normalized = lax.scan(
        body,
        init_carry,
        (sequence, jnp.arange(sequence.shape[0], dtype=jnp.int32)),
    )
    return normalized


def causal_bl_correct(
    X: Float[np.ndarray, "n_epochs n_frames n_win_len n_features"],
    len_scale: int = 100,
    warmup_points: int = 10,
    warmup_scale: int = 100,
    slow_len_scale: int | None = None,
    slow_warmup_scale: int | None = None,
    mean_blend: float = 0.5,
    min_var_fraction: float = 1e-4,
    eps: float = 1e-8,
    device: jax.Device | None = None,
) -> Float[np.ndarray, "n_epochs n_frames n_win_len n_features"]:

    if X.ndim != 4:
        raise ValueError(
            "Input must have shape (n_epochs, n_frames, n_win_len, n_features)."
        )

    len_scale = max(int(len_scale), 1)
    warmup_scale = max(int(warmup_scale), 1)
    warmup_points = max(int(warmup_points), 0)

    slow_len_scale = (
        max(int(slow_len_scale), 1)
        if slow_len_scale is not None
        else max(len_scale * 5, 1)
    )
    slow_warmup_scale = (
        max(int(slow_warmup_scale), 1)
        if slow_warmup_scale is not None
        else max(warmup_scale * 2, 1)
    )

    if not 0.0 <= mean_blend <= 1.0:
        raise ValueError("mean_blend must lie in the interval [0, 1].")
    if min_var_fraction < 0.0:
        raise ValueError("min_var_fraction must be non-negative.")

    data = ensure_device_array(X, device=device)
    device_dtype = jax.dtypes.canonicalize_dtype(data.dtype)

    alpha_main_fast = jnp.asarray(1.0 / len_scale, dtype=device_dtype)
    alpha_warmup_fast = jnp.asarray(1.0 / warmup_scale, dtype=device_dtype)
    alpha_main_slow = jnp.asarray(1.0 / slow_len_scale, dtype=device_dtype)
    alpha_warmup_slow = jnp.asarray(1.0 / slow_warmup_scale, dtype=device_dtype)
    blend_weight = jnp.asarray(mean_blend, dtype=device_dtype)
    min_var_fraction_arr = jnp.asarray(min_var_fraction, dtype=device_dtype)

    def process(series: jnp.ndarray) -> jnp.ndarray:
        if series.shape[0] < 1:
            raise ValueError("Each time series must contain at least one frame.")
        return _causal_normalize(
            series,
            alpha_main_fast=alpha_main_fast,
            alpha_warmup_fast=alpha_warmup_fast,
            alpha_main_slow=alpha_main_slow,
            alpha_warmup_slow=alpha_warmup_slow,
            blend_weight=blend_weight,
            min_var_fraction=min_var_fraction_arr,
            warmup_points=warmup_points,
            eps=eps,
        )

    vmap_features = jax.vmap(process, in_axes=1, out_axes=1)
    vmap_window = jax.vmap(vmap_features, in_axes=1, out_axes=1)
    normalize = jax.vmap(vmap_window, in_axes=0, out_axes=0)

    normalized = normalize(data)
    result = to_numpy(normalized)

    if not np.isfinite(result).all():
        raise ValueError(
            "NaN or Inf detected after causal baseline correction. "
            "Check input data or adjust len_scale parameters."
        )

    return result


def reshape_feats(feats):
    reshaped_data = feats.stack(features=["feature_name", "win_size"])

    # Rename features to include window size
    # Drop relevant dimensions:
    # Ignore FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        features = reshaped_data.features.values
        reshaped_data.drop_vars({"feature_name", "win_size", "features"})

        reshaped_data = reshaped_data.assign_coords(
            {
                "features": [
                    f"{feat_name}_{win_size}" for feat_name, win_size in features
                ]
            }
        )
        reshaped_data = reshaped_data.stack(observation=["epoch", "frame"]).transpose()

    return reshaped_data


@partial(
    jax.jit,
    static_argnames=("frame_length", "hop_length", "initial_offset"),
)
def _create_trailing_frames_jax(
    time_series: Float[Array, "n_trials n_channels n_samples"],
    frame_length: int,
    hop_length: int,
    initial_offset: int,
) -> Float[Array, "n_trials n_frames n_channels n_samples"]:
    start_index = initial_offset - frame_length
    num_frames = (time_series.shape[-1] - initial_offset) // hop_length + 1

    hop = jnp.array(hop_length, dtype=jnp.int32)
    start0 = jnp.array(start_index, dtype=jnp.int32)
    starts = start0 + hop * jnp.arange(num_frames, dtype=jnp.int32)

    def extract(idx: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.dynamic_slice_in_dim(
            time_series,
            idx,
            frame_length,
            axis=2,
        )

    frames = jax.vmap(extract)(starts)
    return jnp.transpose(frames, (1, 0, 2, 3))


def create_trailing_frames(
    time_series: Float[np.ndarray, "n_trials n_channels n_samples"],
    frame_length: int,
    hop_length: int,
    initial_offset: int | None = None,
    device: jax.Device | None = None,
) -> jax.Array:

    assert time_series.ndim == 3, "Time series should be a 3D array"
    n_trials, n_channels, n_samples = time_series.shape

    if initial_offset is None:
        initial_offset = frame_length

    assert frame_length > 0, "Frame length should be a positive integer"
    assert (
        frame_length <= n_samples
    ), "Frame length should be less than or equal to the length of the time series"
    assert hop_length > 0, "Offset should be a positive integer"
    assert (
        initial_offset >= frame_length
    ), "Initial offset should be greater than or equal to the frame length"
    assert (
        initial_offset <= n_samples - hop_length
    ), "Initial offset should be less than the length of the time series"
    assert (
        time_series.ndim == 3
    ), "Time series should be a 3D array of shape `n_trials x n_channels x n_samples`"

    device_dtype = jax.dtypes.canonicalize_dtype(time_series.dtype)
    ts_device = ensure_device_array(time_series, device=device, dtype=device_dtype)

    frames = _create_trailing_frames_jax(
        ts_device,
        frame_length=frame_length,
        hop_length=hop_length,
        initial_offset=initial_offset,
    )

    return frames
