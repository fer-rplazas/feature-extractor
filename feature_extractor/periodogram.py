from __future__ import annotations

from functools import partial
from typing import Sequence, Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .assets import default_canonical_freq_bands
from .utils import ensure_device_array


def _hanning(length: int, dtype: jnp.dtype) -> Array:
    """Custom Hanning window because jnp.hanning is not available in older JAX."""
    if length <= 0:
        raise ValueError("Signal length must be a positive integer.")

    if length == 1:
        return jnp.ones((1,), dtype=dtype)

    indices = jnp.arange(length, dtype=dtype)
    return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * indices / (length - 1))


@partial(jax.jit, static_argnames=("normalize",))
def _extract_features(
    X: Float[Array, "epochs frames channels samples"],
    fs: float,
    band_edges: Float[Array, "bands two"],
    normalization_cutoff: float,
    *,
    normalize: bool,
) -> Float[Array, "epochs frames channels features"]:
    signal_dtype = X.dtype
    n_samples = X.shape[-1]

    window = _hanning(n_samples, signal_dtype)
    windowed = X * window

    freqs = jnp.fft.rfftfreq(n_samples, d=1.0 / fs)
    fft_vals = jnp.fft.rfft(windowed, axis=-1)

    window_power = jnp.sum(window**2) / n_samples
    scale = 1.0 / (window_power * n_samples)
    spectrum = scale * (jnp.abs(fft_vals) ** 2)
    spectrum = jnp.log1p(spectrum + jnp.finfo(signal_dtype).eps)

    lows = band_edges[:, 0]
    highs = band_edges[:, 1]

    def band_mask(lo, hi):
        return (freqs >= lo) & (freqs < hi)

    masks = jax.vmap(band_mask)(lows, highs).astype(signal_dtype)
    counts = jnp.maximum(masks.sum(axis=1), 1.0).astype(signal_dtype)

    band_sum = jnp.einsum("...f,bf->...b", spectrum, masks)
    band_mean = band_sum / counts

    band_sq_sum = jnp.einsum("...f,bf->...b", spectrum**2, masks)
    band_var = jnp.maximum(band_sq_sum / counts - band_mean**2, 0.0)
    band_std = jnp.sqrt(band_var)

    def normalize_means(means):
        norm_mask = (freqs < normalization_cutoff).astype(signal_dtype)
        norm_count = jnp.maximum(norm_mask.sum(), 1.0).astype(signal_dtype)
        total_power = jnp.einsum("...f,f->...", spectrum, norm_mask) / norm_count
        divisor = total_power[..., None] + jnp.array(1e-7, dtype=signal_dtype)
        return means / divisor

    band_mean = jax.lax.cond(normalize, normalize_means, lambda x: x, band_mean)

    return jnp.concatenate([band_mean, band_std], axis=-1)


class PeriodogramFeatureExtractor:
    def __init__(
        self,
        freq_bands: Sequence[tuple[float, float]] | None = None,
        normalize: bool = False,
        normalization_cutoff: float = 120.0,
        device: jax.Device | None = None,
    ) -> Self:
        self.freq_bands = list(freq_bands or default_canonical_freq_bands)
        self.normalize = normalize
        self.normalization_cutoff = normalization_cutoff
        self.device = device

        self._band_edges = np.asarray(self.freq_bands, dtype=np.float32)
        if self._band_edges.ndim != 2 or self._band_edges.shape[1] != 2:
            raise ValueError("Frequency bands must be provided as (low, high) tuples.")

        if np.any(self._band_edges[:, 0] >= self._band_edges[:, 1]):
            raise ValueError("Each frequency band must satisfy low < high.")

        self._compiled = _extract_features

    def feat_names(self, n_channels: int):
        prefix = "PowNorm" if self.normalize else "Pow"
        return [
            f"{prefix}{f_low}-{f_high}_ch{ch}"
            for ch in range(n_channels)
            for f_low, f_high in self.freq_bands
        ] + [
            f"{prefix}Std_{f_low}-{f_high}_ch{ch}"
            for ch in range(n_channels)
            for f_low, f_high in self.freq_bands
        ]

    def get_feats(
        self,
        X: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"],
        fs: float,
    ) -> Float[Array, "n_epochs n_frames n_channels n_features"]:
        if isinstance(X, jax.Array):
            if X.ndim != 4:
                raise ValueError(
                    "Input X must have shape (n_epochs, n_frames, n_channels, n_samples)."
                )
            arr = X
            if not np.issubdtype(np.dtype(arr.dtype), np.floating):
                arr = arr.astype(jnp.float32)
            device_dtype = jax.dtypes.canonicalize_dtype(arr.dtype)
            X_device = ensure_device_array(arr, device=self.device, dtype=device_dtype)
            n_samples = arr.shape[-1]
        else:
            X_np = np.asarray(X)
            if X_np.ndim != 4:
                raise ValueError(
                    "Input X must have shape (n_epochs, n_frames, n_channels, n_samples)."
                )
            if not np.issubdtype(X_np.dtype, np.floating):
                X_np = X_np.astype(np.float32)
            device_dtype = jax.dtypes.canonicalize_dtype(X_np.dtype)
            X_device = ensure_device_array(X_np, device=self.device, dtype=device_dtype)
            n_samples = X_np.shape[-1]

        if fs <= 0:
            raise ValueError("Sampling frequency must be positive.")

        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        band_masks = (freqs[None, :] >= self._band_edges[:, :1]) & (
            freqs[None, :] < self._band_edges[:, 1:]
        )

        if np.any(band_masks.sum(axis=1) == 0):
            raise ValueError(
                "Each frequency band must cover at least one spectral bin; "
                "consider relaxing the band edges or increasing signal length."
            )

        band_edges = ensure_device_array(
            self._band_edges, device=self.device, dtype=device_dtype
        )

        feats = self._compiled(
            X_device,
            float(fs),
            band_edges,
            float(self.normalization_cutoff),
            normalize=self.normalize,
        )
        if bool(jnp.any(jnp.isnan(feats))):
            raise ValueError(
                "NaN values found in features during periodogram-based feature "
                "extraction. Check frequency band indices."
            )

        return feats
