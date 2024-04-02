import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from jaxtyping import Float


class PSIFeatureExtractor:
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
        return [f"{f_low}-{f_high}" for f_low, f_high in self.freq_bands]

    def get_feats(
        self,
        X1: Float[np.ndarray, "n_epochs n_frames n_samples"],
        X2: Float[np.ndarray, "n_epochs n_frames n_samples"],
        fs: float = 2048.0,
    ):
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"

        n_samples = X1.shape[-1]
        window = np.hanning(n_samples)

        X1 *= window
        X2 *= window

        feats = np.empty((X1.shape[0], X1.shape[1], len(self.freq_bands)))

        for band_idx, (f_low, f_high) in enumerate(self.freq_bands):
            # Bandpass filtering
            sos = butter(3, [f_low, f_high], btype="band", fs=fs, output="sos")
            filtered_X1 = sosfiltfilt(sos, X1, axis=-1)
            filtered_X2 = sosfiltfilt(sos, X2, axis=-1)

            # Compute the analytic signal
            analytic_signal1 = hilbert(filtered_X1, axis=-1)
            analytic_signal2 = hilbert(filtered_X2, axis=-1)

            # Compute the phase of the analytic signals
            phase1 = np.angle(analytic_signal1)
            phase2 = np.angle(analytic_signal2)

            # Compute the phase synchrony index (PSI)
            psi = np.abs(np.mean(np.exp(np.complex(0, 1) * (phase1 - phase2)), axis=-1))

            # Store the PSI for each frequency band
            feats[:, :, band_idx] = psi

        return feats
