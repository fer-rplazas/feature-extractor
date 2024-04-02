import numpy as np
from jaxtyping import Float
from numba import njit


class CoherenceFeatureExtractor:
    def __init__(
        self,
        freq_bands: list[tuple[float, float]] = [
            (4, 8),
            (8, 12),
            (13, 30),
            (30, 50),
            (50, 70),
            (70, 100),
            (100, 200),
        ],
    ):
        self.freq_bands = freq_bands

    def feat_names(self):
        return [
            f"{mode}{f_low}-{f_high}"
            for f_low, f_high in self.freq_bands
            for mode in ["Coh", "iCoh"]
        ] + ["CohAll", "iCohAll"]

    def get_feats(
        self,
        X1: Float[np.ndarray, "n_epochs n_frames n_samples"],
        X2: Float[np.ndarray, "n_epochs n_frames n_samples"],
        fs: float = 2048.0,
    ) -> Float[np.ndarray, "n_epochs n_frames 2*n_bands+2"]:
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"

        n_samples = X1.shape[-1]
        window = np.hanning(n_samples)

        X1 = X1 * window
        X2 = X2 * window

        # Initialize the output array
        feats = np.empty((X1.shape[0], X1.shape[1], len(self.freq_bands) * 2 + 2))

        X1_fft = np.fft.fft(X1, axis=-1)
        X2_fft = np.fft.fft(X2, axis=-1)

        freqs = np.fft.fftfreq(n_samples, d=1 / fs)

        cross_spectral_density = X1_fft * np.conj(X2_fft)
        X1_auto_spectrum = X1_fft * np.conj(X1_fft)
        X2_auto_spectrum = X2_fft * np.conj(X2_fft)

        coherence = np.abs(
            cross_spectral_density**2 / (X1_auto_spectrum * X2_auto_spectrum) + 1e-7
        )
        icoherence = np.abs(
            np.imag(cross_spectral_density)
            / (np.sqrt(X1_auto_spectrum * X2_auto_spectrum) + 1e-7)
        )

        # Compute coherences without frequency filtering
        feats[:, :, -2:] = np.stack(
            (np.mean(coherence, axis=-1), np.mean(icoherence, axis=-1)), axis=-1
        )

        # Compute coherences for each frequency band
        for idx, band in enumerate(self.freq_bands):
            freq_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
            feats[:, :, 2 * idx] = np.mean(coherence[..., freq_idx], axis=-1)
            feats[:, :, 2 * idx + 1] = np.mean(icoherence[..., freq_idx], axis=-1)

        return feats
