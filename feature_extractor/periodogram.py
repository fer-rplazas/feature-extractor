import numpy as np
from jaxtyping import Float
from numba import njit, prange
from scipy.signal import periodogram

from .utils import array_idx
from .assets import default_canonical_freq_bands


@njit(parallel=False)
def calculate_features(Pxx, f_lows, f_highs, w_120, normalize=False):
    n_bands = len(f_lows)
    feats = np.zeros((Pxx.shape[0], Pxx.shape[1], Pxx.shape[2], n_bands * 2))
    for kk in prange(n_bands):
        f_low, f_high = f_lows[kk], f_highs[kk]
        mean_val = np.sum(Pxx[..., f_low:f_high], axis=-1) / (f_high - f_low)
        feats[..., kk] = mean_val

        sum_sq_diff = np.sum(
            (Pxx[..., f_low:f_high] - mean_val[..., np.newaxis]) ** 2, axis=-1
        )
        feats[..., n_bands + kk] = np.sqrt(sum_sq_diff / (f_high - f_low))

        if normalize:
            total_power = np.sum(Pxx[..., :w_120], axis=-1) / w_120
            feats[..., kk] = feats[..., kk] / (total_power + 1e-7)
    return feats


class PeriodogramFeatureExtractor:
    def __init__(
        self,
        freq_bands: list[tuple[float, float]] | None = None,
        normalize: bool = False,
    ):
        self.freq_bands = freq_bands or default_canonical_freq_bands

        self.normalize = normalize

    def feat_names(self, n_channels: int):
        if self.normalize:
            prefix = "PowNorm"
        else:
            prefix = "Pow"

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
        fs: float = 2048.0,
    ) -> Float[np.ndarray, "n_epochs n_frames n_channels n_bands"]:
        n_samples = X.shape[-1]
        window = np.hanning(n_samples)

        w_ref, Pxx = periodogram(X * window, fs=fs, axis=-1)

        # Log-scale the periodogram
        Pxx = np.log1p(Pxx + 1e-7)

        idx = [array_idx(w_ref, el) for el in self.freq_bands]
        w_120, _ = array_idx(w_ref, (120, 140))

        f_lows = np.array([f_low for f_low, _ in idx])
        f_highs = np.array([f_high for _, f_high in idx])

        feats = calculate_features(
            Pxx, f_lows, f_highs, w_120, normalize=self.normalize
        )

        return feats
