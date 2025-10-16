from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .utils import ensure_device_array


def _hanning_window(length: int, dtype: jnp.dtype) -> Array:
    if length <= 0:
        raise ValueError("Signal length must be positive.")
    if length == 1:
        return jnp.ones((1,), dtype=dtype)
    indices = jnp.arange(length, dtype=dtype)
    return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * indices / (length - 1))


def _compute_band_edges(n_ceps: int, n_bands: int) -> np.ndarray:
    n_half = int(n_ceps // 2)
    edges = [1]
    for idx in range(1, n_bands + 1):
        edge = int(n_half ** (idx / n_bands))
        edges.append(min(edge, n_half))
    return np.asarray(edges, dtype=np.int32)


@jax.jit
def _cepstrum_bands(
    signal: Float[Array, "epochs frames channels samples"],
    band_masks: Float[Array, "bands samples"],
) -> Float[Array, "epochs frames channels features"]:
    out_dtype = signal.dtype
    compute_dtype = jnp.float32

    signal = signal.astype(compute_dtype)
    band_masks = band_masks.astype(compute_dtype)

    n_samples = signal.shape[-1]

    window = _hanning_window(n_samples, compute_dtype)
    windowed = signal * window

    spectrum = jnp.fft.rfft(windowed, axis=-1)
    log_magnitude = jnp.log(jnp.abs(spectrum) + jnp.finfo(compute_dtype).eps)
    cepstrum = jnp.fft.irfft(log_magnitude, n=n_samples, axis=-1)

    counts = band_masks.sum(axis=1).astype(compute_dtype)
    counts_safe = jnp.where(counts > 0, counts, 1.0)

    means = jnp.einsum("...s,bs->...b", cepstrum, band_masks) / counts_safe
    second_mom = jnp.einsum("...s,bs->...b", cepstrum**2, band_masks) / counts_safe
    vars_ = jnp.maximum(second_mom - means**2, 0.0)
    stds = jnp.sqrt(vars_)

    means = jnp.where(counts > 0, means, 0.0)
    stds = jnp.where(counts > 0, stds, 0.0)

    feats = jnp.concatenate([means, stds], axis=-1)
    return feats.astype(out_dtype)


class CepstralFeatureExtractor:
    def __init__(self, n_bands: int = 8, device: jax.Device | None = None):
        self.n_bands = n_bands
        self.n_coef = n_bands * 2
        self.device = device

    def feat_names(self, n_channels: int):
        return [
            f"CEP{band}{suffix}_ch{ch}"
            for ch in range(n_channels)
            for band in range(self.n_bands)
            for suffix in ("-mean", "-std")
        ]

    def get_feats(
        self,
        X: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"],
    ) -> Float[Array, "n_epochs n_frames n_channels n_features"]:
        if isinstance(X, jax.Array):
            if X.ndim != 4:
                raise ValueError(
                    "Input must have shape (n_epochs, n_frames, n_channels, n_samples)."
                )

            arr = X
            if not np.issubdtype(np.dtype(arr.dtype), np.floating):
                arr = arr.astype(jnp.float32)
            device_dtype = jax.dtypes.canonicalize_dtype(arr.dtype)
            signal = ensure_device_array(arr, device=self.device, dtype=device_dtype)
            n_samples = arr.shape[-1]
        else:
            X_np = np.asarray(X)
            if X_np.ndim != 4:
                raise ValueError(
                    "Input must have shape (n_epochs, n_frames, n_channels, n_samples)."
                )

            if not np.issubdtype(X_np.dtype, np.floating):
                X_np = X_np.astype(np.float32)

            device_dtype = jax.dtypes.canonicalize_dtype(X_np.dtype)
            signal = ensure_device_array(X_np, device=self.device, dtype=device_dtype)
            n_samples = X_np.shape[-1]

        band_edges = _compute_band_edges(n_samples, self.n_bands)

        mask = np.zeros((self.n_bands, n_samples), dtype=np.float32)
        for idx in range(self.n_bands):
            low = band_edges[idx]
            high = band_edges[idx + 1]
            mask[idx, low:high] = 1.0

        band_masks = ensure_device_array(mask, device=self.device, dtype=device_dtype)

        stats = _cepstrum_bands(signal, band_masks)

        if bool(jnp.any(jnp.isnan(stats))):
            raise ValueError(
                "NaN or Inf found in cepstral features; verify input data."
            )

        return stats
