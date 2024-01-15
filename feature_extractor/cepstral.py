import numpy as np
from numba import njit, prange


def compute_cepstrum(signal):
    """
    Compute the real cepstrum of a 1D signal.

    :param signal: Input signal (1D numpy array).
    :param n_coeffs: Number of cepstral coefficients to return. If None, returns all coefficients.
    :return: Cepstral coefficients (1D numpy array).
    """
    # Compute the Fourier Transform
    spectrum = np.fft.fft(signal)

    # Compute the log magnitude spectrum
    log_magnitude = np.log(
        np.abs(spectrum) + np.finfo(float).eps
    )  # Add eps for numerical stability

    # Compute the inverse Fourier Transform
    cepstrum = np.fft.ifft(log_magnitude).real

    # Return the number of requested coefficients
    return cepstrum


@njit(parallel=True)
def fill_feats(cepstrum, band_edges, feats):
    n_epochs, n_frames, n_channels, _ = cepstrum.shape
    n_bands = len(band_edges) - 1

    for i in prange(n_epochs):
        for j in prange(n_frames):
            for k in prange(n_channels):
                for b in prange(n_bands):
                    low_idx = band_edges[b]
                    high_idx = band_edges[b + 1]

                    # Ensure indices are valid
                    if low_idx < high_idx:
                        band_ceps = cepstrum[i, j, k, low_idx:high_idx]
                        mean = np.mean(band_ceps)
                        std = np.std(band_ceps)
                    else:
                        mean = 0.0
                        std = 0.0

                    feats[i, j, k, b * 2] = mean  # Mean of the band
                    feats[i, j, k, b * 2 + 1] = std  # Std of the band


class CepstralFeatureExtractor:
    def __init__(
        self,
        n_bands: int = 8,
    ):
        self.n_bands = n_bands
        self.n_coef = n_bands * 2

    def feat_names(self, n_channels: int):
        return [
            f"CEP{coef}{mode}_ch{ch}"
            for ch in range(n_channels)
            for coef in range(self.n_bands)
            for mode in ["-mean", "-std"]
        ]

    def _compute_band_edges(self, n_ceps: int):
        n_ceps = int(n_ceps / 2)  # Length of the first half of the cepstrum
        band_edges = [1]  # Start from index 1 to exclude the zeroth quefrency

        # Calculate exponentially increasing indices
        for i in range(1, self.n_bands + 1):
            edge = int(n_ceps ** (i / self.n_bands))
            band_edges.append(min(edge, n_ceps))  # Ensure edge does not exceed n_ceps

        return band_edges

    def get_feats(self, X):
        # Compute cepstrum for the entire array
        n_samples = X.shape[-1]
        window = np.hanning(n_samples)
        cepstrum = compute_cepstrum(X * window)

        # Compute band edges
        band_edges = self._compute_band_edges(cepstrum.shape[-1])

        # Initialize an empty array for features
        feats = np.empty(
            (X.shape[0], X.shape[1], X.shape[2], self.n_coef)
        )  # times 2 for mean and std

        # Fill the feats matrix using the Numba-accelerated function
        fill_feats(cepstrum, np.array(band_edges, dtype=int), feats)

        return feats
