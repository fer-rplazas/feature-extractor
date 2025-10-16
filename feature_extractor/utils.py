from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erfinv
import numpy as np
from jaxtyping import Float

import warnings


def ensure_device_array(
    value: Any,
    *,
    device: jax.Device | None = None,
    dtype: jnp.dtype | None = None,
) -> jax.Array:
    if isinstance(value, jax.Array):
        arr = value
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        if device is not None:
            arr = jax.device_put(arr, device)
        return arr

    np_value = np.asarray(value)
    if dtype is not None:
        np_value = np_value.astype(dtype, copy=False)
    return jax.device_put(np_value, device)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, jax.Array):
        return np.asarray(jax.device_get(value))
    return np.asarray(value)


def _winsor_scale_from_limit(limit: float, dtype: jnp.dtype) -> jnp.ndarray:
    limit = jnp.asarray(limit, dtype=dtype)
    limit = jnp.clip(limit, 1e-6, 0.499999)
    # Convert percentile to number of standard deviations assuming a Gaussian.
    sqrt_two = jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    return sqrt_two * erfinv(jnp.asarray(2.0 * (1.0 - limit) - 1.0, dtype=dtype))


def _causal_winsor_normalize(
    sequence: jnp.ndarray,
    alpha_main: jnp.ndarray,
    alpha_warmup: jnp.ndarray,
    warmup_points: int,
    winsor_scale: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    if sequence.ndim != 1:
        raise ValueError("Expected 1D sequence for causal normalization.")

    def body(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ):
        mean, var = carry
        value, idx = inputs

        std = jnp.sqrt(jnp.maximum(var, eps))
        lower = mean - winsor_scale * std
        upper = mean + winsor_scale * std
        clipped = jnp.clip(value, lower, upper)

        alpha = jnp.where(idx < warmup_points, alpha_warmup, alpha_main)
        delta = clipped - mean
        mean = mean + alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta**2)
        std_new = jnp.sqrt(jnp.maximum(var, eps))
        normalized = (clipped - mean) / (std_new + eps)

        return (mean, var), normalized

    init_mean = sequence[0]
    init_var = jnp.zeros_like(init_mean)

    _, normalized = lax.scan(
        body,
        (init_mean, init_var),
        (sequence, jnp.arange(sequence.shape[0], dtype=jnp.int32)),
    )
    return normalized


def causal_bl_correct(
    X: Float[np.ndarray, "n_epochs n_frames n_win_len n_features"],
    winsor_limit: float = 0.02,
    len_scale: int = 100,
    warmup_points: int = 10,
    warmup_scale: int = 100,
    eps: float = 1e-8,
    device: jax.Device | None = None,
) -> Float[np.ndarray, "n_epochs n_frames n_win_len n_features"]:
    X_np = np.asarray(X)
    if X_np.ndim != 4:
        raise ValueError(
            "Input must have shape (n_epochs, n_frames, n_win_len, n_features)."
        )

    if not np.issubdtype(X_np.dtype, np.floating):
        X_np = X_np.astype(np.float32)

    if not 0.0 < winsor_limit < 0.5:
        raise ValueError("winsor_limit must be between 0 and 0.5.")

    len_scale = max(int(len_scale), 1)
    warmup_scale = max(int(warmup_scale), 1)
    warmup_points = max(int(warmup_points), 0)

    device_dtype = jax.dtypes.canonicalize_dtype(X_np.dtype)
    data = ensure_device_array(X_np, device=device, dtype=device_dtype)

    alpha_main = jnp.asarray(1.0 / len_scale, dtype=device_dtype)
    alpha_warmup = jnp.asarray(1.0 / warmup_scale, dtype=device_dtype)
    winsor_scale = _winsor_scale_from_limit(winsor_limit, device_dtype)

    def process(series: jnp.ndarray) -> jnp.ndarray:
        if series.shape[0] < 1:
            raise ValueError("Each time series must contain at least one frame.")
        return _causal_winsor_normalize(
            series,
            alpha_main=alpha_main,
            alpha_warmup=alpha_warmup,
            warmup_points=warmup_points,
            winsor_scale=winsor_scale,
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
            "Check input data or adjust winsor/len_scale parameters."
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
    time_series: jnp.ndarray,
    frame_length: int,
    hop_length: int,
    initial_offset: int,
) -> jnp.ndarray:
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
