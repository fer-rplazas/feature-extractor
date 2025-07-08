import numpy as np
from scipy.signal import coherence, csd, get_window
from jaxtyping import Float
from typing import List, Tuple, Optional
from joblib import Parallel, delayed

from .assets import default_canonical_freq_bands


class CoherenceFeatureExtractor:
    def __init__(
        self,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        window_type: str = "hann",
        scaling: str = "density",
        average: str = "mean",
        n_jobs: int = -1,  # Number of parallel jobs (-1 uses all available cores)
    ):
        """
        Initializes the Coherence Feature Extractor.

        Parameters:
        - freq_bands (Optional[List[Tuple[float, float]]]):
            List of tuples specifying frequency bands (low, high).
            Defaults to canonical bands if None.
        - window_type (str):
            Desired window to use for segmenting the signal. Default is 'hann'.
        - scaling (str):
            Selects between 'density' and 'spectrum'. 'density' is appropriate for coherence.
        - average (str):
            How to average the coherence estimates within each band. Default is 'mean'.
        - n_jobs (int):
            Number of parallel jobs for computation. Default is -1 (all cores).
        """
        self.freq_bands = freq_bands or default_canonical_freq_bands
        self.window_type = window_type
        self.scaling = scaling
        self.average = average
        self.n_jobs = n_jobs

    def feat_names(self) -> List[str]:
        """
        Generates descriptive feature names based on frequency bands.

        Returns:
            List[str]: A list of feature names in the format "CohLow-HighHz", "iCohLow-HighHz",
                       followed by "CohAll" and "iCohAll".
        """
        band_names = [f"Coh{f_low}-{f_high}Hz" for f_low, f_high in self.freq_bands] + [
            f"iCoh{f_low}-{f_high}Hz" for f_low, f_high in self.freq_bands
        ]
        # Add overall coherence and imaginary coherence
        # band_names += ["CohAll", "iCohAll"]
        return band_names

    def _compute_frame_feats(
        self, frame_X1: np.ndarray, frame_X2: np.ndarray, fs: float
    ) -> np.ndarray:
        """
        Computes coherence features for a single frame.

        Parameters:
            frame_X1 (np.ndarray): Signal 1 for the frame with shape (n_samples,).
            frame_X2 (np.ndarray): Signal 2 for the frame with shape (n_samples,).
            fs (float): Sampling frequency in Hz.

        Returns:
            np.ndarray: Coherence features for the frame with shape (2*n_bands + 2,).
        """
        n_samples = frame_X1.shape[0]
        n_bands = len(self.freq_bands)

        # Dynamically set nperseg and noverlap based on frame length
        nperseg = max(256, n_samples // 3)  # Ensure nperseg is at least 256
        noverlap = nperseg // 2  # 50% overlap

        # Adjust nperseg if it's larger than n_samples
        if nperseg > n_samples:
            nperseg = n_samples
            noverlap = nperseg // 2

        # Compute coherence using Welch's method
        f, Cxy = coherence(
            frame_X1,
            frame_X2,
            fs=fs,
            window=self.window_type,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # Compute cross-spectral density using csd for imaginary coherence
        f_csd, Pxy = csd(
            frame_X1,
            frame_X2,
            fs=fs,
            window=self.window_type,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling=self.scaling,
        )

        # Compute imaginary coherence
        icoherence = np.abs(np.imag(Pxy)) / (
            np.sqrt(np.abs(Pxy.real) ** 2 + np.abs(Pxy.imag) ** 2) + 1e-10
        )

        # Compute overall coherence and imaginary coherence across all frequencies
        # coh_all_final = np.mean(Cxy)
        # icoh_all_final = np.mean(icoherence)

        # Initialize lists to hold band-specific coherence
        coh_bands = []
        icoh_bands = []

        for f_low, f_high in self.freq_bands:
            # Find frequency indices within the current band
            band_idx_mask = (f >= f_low) & (f <= f_high)
            if not np.any(band_idx_mask):
                raise ValueError(
                    f"No frequency bins found for the band {f_low}-{f_high}Hz."
                )

            # Compute mean coherence within the band
            coh_band = np.mean(Cxy[band_idx_mask])
            coh_bands.append(coh_band)

            # Compute mean imaginary coherence within the band
            icoh_band = np.mean(icoherence[band_idx_mask])
            icoh_bands.append(icoh_band)

        # Assemble features
        features = coh_bands + icoh_bands  # + [coh_all_final, icoh_all_final]

        return np.array(features, dtype=np.float64)

    def _compute_epoch_feats(
        self, epoch_X1: np.ndarray, epoch_X2: np.ndarray, fs: float
    ) -> np.ndarray:
        """
        Computes coherence features for a single epoch across all frames.

        Parameters:
            epoch_X1 (np.ndarray): Signal 1 for the epoch with shape (n_frames, n_samples).
            epoch_X2 (np.ndarray): Signal 2 for the epoch with shape (n_frames, n_samples).
            fs (float): Sampling frequency in Hz.

        Returns:
            np.ndarray: Coherence features for the epoch with shape (n_frames, 2*n_bands + 2).
        """
        n_frames, n_samples = epoch_X1.shape
        n_bands = len(self.freq_bands)

        # Initialize array to hold features for all frames in the epoch
        epoch_feats = np.empty((n_frames, 2 * n_bands), dtype=np.float64)

        for frame_idx in range(n_frames):
            frame_X1 = epoch_X1[frame_idx, :]
            frame_X2 = epoch_X2[frame_idx, :]
            epoch_feats[frame_idx, :] = self._compute_frame_feats(
                frame_X1, frame_X2, fs
            )

        return epoch_feats

    def get_feats(
        self,
        X1: Float[np.ndarray, "n_epochs n_frames n_samples"],
        X2: Float[np.ndarray, "n_epochs n_frames n_samples"],
        fs: float = 2048.0,
    ) -> Float[np.ndarray, "n_epochs n_frames 2*n_bands"]:
        """
        Computes the Coherence and Imaginary Coherence between two signals across specified frequency bands for each frame.

        Parameters:
            X1 (np.ndarray): Signal 1 with shape (n_epochs, n_frames, n_samples).
            X2 (np.ndarray): Signal 2 with shape (n_epochs, n_frames, n_samples).
            fs (float): Sampling frequency in Hz. Default is 2048.0.

        Returns:
            np.ndarray: Coherence features with shape (n_epochs, n_frames, 2*n_freq_bands + 2),
                        where each frequency band contributes "Coh" and "iCoh" features,
                        followed by "CohAll" and "iCohAll".
        """
        assert X1.shape == X2.shape, "X1 and X2 must have the same shape"
        n_epochs, n_frames, n_samples = X1.shape
        n_bands = len(self.freq_bands)

        # Define a helper function for parallel processing per epoch
        def process_epoch(epoch_idx):
            epoch_X1 = X1[epoch_idx, :, :]  # Shape: (n_frames, n_samples)
            epoch_X2 = X2[epoch_idx, :, :]  # Shape: (n_frames, n_samples)
            return self._compute_epoch_feats(epoch_X1, epoch_X2, fs)

        # Use joblib for parallel processing across epochs
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_epoch)(epoch_idx) for epoch_idx in range(n_epochs)
        )

        # Stack results into a NumPy array with shape (n_epochs, n_frames, 2*n_bands + 2)
        feats = np.stack(results, axis=0)

        return feats
