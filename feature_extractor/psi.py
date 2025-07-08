import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from jaxtyping import Float

from .assets import default_canonical_freq_bands


class PSIFeatureExtractor:
    def __init__(
        self,
        freq_bands: list[tuple[float, float]] | None = None,
        filter_order: int = 3,
        pad_size: int = 128,
        pad_mode: str = "reflect",
    ):
        """
        Initializes the PSI Feature Extractor.

        Parameters:
        - freq_bands: List of tuples specifying frequency bands (low, high).
                      Defaults to canonical bands if None.
        - filter_order: Order of the Butterworth bandpass filter. Default is 3.
        - pad_size: Number of samples to pad on each side of the signal to mitigate edge artifacts.
                    Default is 100 samples.
        - pad_mode: Padding mode to use. Common options include 'reflect', 'constant', etc.
                    Default is 'reflect'.
        """
        self.freq_bands = freq_bands or default_canonical_freq_bands
        self.filter_order = filter_order
        self.pad_size = pad_size
        self.pad_mode = pad_mode

    def feat_names(self):
        """
        Generates descriptive feature names based on frequency bands.

        Returns:
            List[str]: A list of feature names in the format "low-highHz".
        """
        return [f"{f_low}-{f_high}Hz" for f_low, f_high in self.freq_bands]

    def get_feats(
        self,
        X1: Float[np.ndarray, "n_epochs n_frames n_samples"],
        X2: Float[np.ndarray, "n_epochs n_frames n_samples"],
        fs: float = 2048.0,
    ) -> np.ndarray:
        """
        Computes the Phase Synchrony Index (PSI) between two signals across specified frequency bands.

        Parameters:
            X1 (np.ndarray): Signal 1 with shape (n_epochs, n_frames, n_samples).
            X2 (np.ndarray): Signal 2 with shape (n_epochs, n_frames, n_samples).
            fs (float): Sampling frequency in Hz. Default is 2048.0.

        Returns:
            np.ndarray: PSI features with shape (n_epochs, n_frames, n_freq_bands).
        """
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"

        n_epochs, n_frames, n_samples = X1.shape

        # Apply windowing
        window = np.hanning(n_samples)
        X1_windowed = X1 * window
        X2_windowed = X2 * window

        # Apply padding to mitigate edge artifacts
        pad_width = ((0, 0), (0, 0), (self.pad_size, self.pad_size))
        X1_padded = np.pad(X1_windowed, pad_width=pad_width, mode=self.pad_mode)
        X2_padded = np.pad(X2_windowed, pad_width=pad_width, mode=self.pad_mode)

        # Initialize feature array
        n_freq_bands = len(self.freq_bands)
        feats = np.empty((n_epochs, n_frames, n_freq_bands), dtype=np.float32)

        for band_idx, (f_low, f_high) in enumerate(self.freq_bands):
            # Design bandpass filter
            sos = butter(
                self.filter_order,
                [f_low, f_high],
                btype="band",
                fs=fs,
                output="sos",
            )

            # Bandpass filter the padded signals
            filtered_X1 = sosfiltfilt(sos, X1_padded, axis=-1)
            filtered_X2 = sosfiltfilt(sos, X2_padded, axis=-1)

            # Remove padding after filtering
            filtered_X1 = filtered_X1[..., self.pad_size : -self.pad_size]
            filtered_X2 = filtered_X2[..., self.pad_size : -self.pad_size]

            # Compute the analytic signal using the Hilbert transform
            analytic_signal1 = hilbert(filtered_X1, axis=-1)
            analytic_signal2 = hilbert(filtered_X2, axis=-1)

            # Extract the instantaneous phase
            phase1 = np.angle(analytic_signal1)
            phase2 = np.angle(analytic_signal2)

            # Compute the Phase Synchrony Index (PSI)
            phase_diff = phase1 - phase2
            sin_sq = np.sin(phase_diff) ** 2
            cos_sq = np.cos(phase_diff) ** 2
            psi = 1 - np.sqrt(
                np.mean(sin_sq, axis=-1)
                / (np.mean(cos_sq, axis=-1) + np.mean(sin_sq, axis=-1) + 1e-8)
            )

            # Store the PSI values for the current frequency band
            feats[:, :, band_idx] = psi.astype(np.float32)

        return feats
