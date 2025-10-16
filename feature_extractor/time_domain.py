from __future__ import annotations

from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .utils import ensure_device_array, to_numpy


def _make_hjorth(
    signal: Float[Array, "epochs frames channels samples"],
    eps: float,
) -> Float[Array, "epochs frames channels metrics"]:
    n_samples = signal.shape[-1]

    if n_samples > 1:
        first_deriv = jnp.diff(signal, axis=-1)
    else:
        first_deriv = jnp.zeros(signal.shape[:-1] + (0,), dtype=signal.dtype)

    if first_deriv.shape[-1] > 1:
        second_deriv = jnp.diff(first_deriv, axis=-1)
    else:
        second_deriv = jnp.zeros(first_deriv.shape[:-1] + (0,), dtype=signal.dtype)

    activity = jnp.maximum(jnp.var(signal, axis=-1), 0.0)

    if first_deriv.shape[-1] > 0:
        var_d1 = jnp.maximum(jnp.var(first_deriv, axis=-1), 0.0)
    else:
        var_d1 = jnp.zeros(signal.shape[:-1], dtype=signal.dtype)

    if second_deriv.shape[-1] > 0:
        var_d2 = jnp.maximum(jnp.var(second_deriv, axis=-1), 0.0)
    else:
        var_d2 = jnp.zeros(signal.shape[:-1], dtype=signal.dtype)

    mobility = jnp.sqrt(var_d1 / (activity + eps))
    complexity = jnp.sqrt(var_d2 / (var_d1 + eps)) / (mobility + eps)

    return jnp.stack((activity, mobility, complexity), axis=-1)


@jax.jit
def _time_domain_features(
    signal: Float[Array, "epochs frames channels samples"],
    eps: float,
) -> Float[Array, "epochs frames channels features"]:
    mean = jnp.mean(signal, axis=-1, keepdims=True)
    energy = jnp.mean(jnp.square(signal), axis=-1, keepdims=True)

    centered = signal - mean
    variance = jnp.maximum(jnp.var(signal, axis=-1), 0.0)
    skew_numer = jnp.mean(centered**3, axis=-1)
    kurt_numer = jnp.mean(centered**4, axis=-1)

    variance_safe = variance + eps
    skew = jnp.where(
        variance > 0,
        skew_numer / jnp.power(variance_safe, 1.5),
        0.0,
    )[..., None]
    kurtosis = jnp.where(
        variance > 0,
        kurt_numer / jnp.square(variance_safe) - 3.0,
        0.0,
    )[..., None]

    hjorth_feats = _make_hjorth(signal, eps)

    return jnp.concatenate(
        [mean, energy, skew, kurtosis, hjorth_feats], axis=-1
    )


def hjorth(
    input_signal: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"],
    eps: float = np.finfo(float).eps,
    device: jax.Device | None = None,
) -> Float[np.ndarray, "n_epochs n_frames n_channels 3"]:
    X_np = np.asarray(input_signal)
    if X_np.ndim != 4:
        raise ValueError(
            "Input signal must have shape "
            "(n_epochs, n_frames, n_channels, n_samples)."
        )
    if X_np.shape[-1] < 1:
        raise ValueError("Input signals must contain at least one sample.")

    if not np.issubdtype(X_np.dtype, np.floating):
        X_np = X_np.astype(np.float32)

    device_dtype = jax.dtypes.canonicalize_dtype(X_np.dtype)
    X_device = ensure_device_array(X_np, device=device, dtype=device_dtype)

    feats = _make_hjorth(X_device, float(eps))
    return to_numpy(feats)


class TimeDomainFeatureExtractor:
    def __init__(
        self,
        eps: float | None = None,
        device: jax.Device | None = None,
    ) -> Self:
        self.eps = eps if eps is not None else float(np.finfo(np.float32).eps)
        self._compiled = _time_domain_features
        self.device = device

    def feat_names(self, n_channels: int):
        return [
            f"{feat_name}_ch{ch}"
            for ch in range(n_channels)
            for feat_name in ("mean", "energy", "skew", "kurt", "act", "mob", "complex")
        ]

    def get_feats(
        self,
        X: Float[np.ndarray, "n_epochs n_frames n_chan n_samples"],
    ) -> Float[Array, "n_epochs n_frames n_chan n_features"]:
        if isinstance(X, jax.Array):
            if X.ndim != 4:
                raise ValueError(
                    "Input X must have shape (n_epochs, n_frames, n_channels, n_samples)."
                )
            if X.shape[-1] < 1:
                raise ValueError("Input signals must contain at least one sample.")

            arr = X
            if not np.issubdtype(np.dtype(arr.dtype), np.floating):
                arr = arr.astype(jnp.float32)
            device_dtype = jax.dtypes.canonicalize_dtype(arr.dtype)
            X_device = ensure_device_array(arr, device=self.device, dtype=device_dtype)
        else:
            X_np = np.asarray(X)
            if X_np.ndim != 4:
                raise ValueError(
                    "Input X must have shape (n_epochs, n_frames, n_channels, n_samples)."
                )
            if X_np.shape[-1] < 1:
                raise ValueError("Input signals must contain at least one sample.")

            if not np.issubdtype(X_np.dtype, np.floating):
                X_np = X_np.astype(np.float32)

            device_dtype = jax.dtypes.canonicalize_dtype(X_np.dtype)
            X_device = ensure_device_array(X_np, device=self.device, dtype=device_dtype)

        feats = self._compiled(X_device, self.eps)

        if bool(jnp.any(jnp.isnan(feats))):
            raise ValueError(
                "NaN values found in time-domain feature extraction; "
                "check the input signal for degenerate cases."
            )

        return feats
