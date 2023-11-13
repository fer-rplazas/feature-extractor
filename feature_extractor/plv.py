import numpy as np
from scipy.signal import hilbert, sosfiltfilt, butter
from jaxtyping import Float


class PLVFeatureExtractor:
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
    ) -> Float[np.ndarray, "n_epochs n_frames n_bands+2"]:
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"

        n_samples = X1.shape[-1]
        window = np.hanning(n_samples)

        X1 = X1 * window
        X2 = X2 * window

        # Initialize the output array
        feats = np.empty((X1.shape[0], X1.shape[1], len(self.freq_bands)))

        for band_idx, (f_low, f_high) in enumerate(self.freq_bands):
            # Create bandpass filter for the current band
            sos = butter(3, [f_low, f_high], btype="band", fs=fs, output="sos")
            # Apply filter to both signals
            filtered_X1 = sosfiltfilt(sos, X1, axis=-1)
            filtered_X2 = sosfiltfilt(sos, X2, axis=-1)

            # Get the phase of the two filtered signals
            analytic_signal1 = hilbert(filtered_X1)
            analytic_signal2 = hilbert(filtered_X2)
            phase1 = np.angle(analytic_signal1)
            phase2 = np.angle(analytic_signal2)

            # Compute phase difference
            phase_diff = phase1 - phase2

            # Calculate the PLV
            plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))

            # Save PLV in output array
            feats[:, :, band_idx] = plv

        return feats
