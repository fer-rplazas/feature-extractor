import dask.array as da
import numpy as np
from dask import compute, delayed
from jaxtyping import Float
from scipy.linalg import solve_toeplitz


def fft_based_lpc(signal, order):
    """
    Compute LPC coefficients using FFT for autocorrelation estimation.

    Parameters:
    signal: 1D array-like, the time series data.
    order: int, the order of the LPC.

    Returns:
    lpc_coeffs: 1D numpy array, LPC coefficients.
    """
    # Ensure the order is less than the length of the signal
    if order >= len(signal):
        raise ValueError("LPC order must be less than the length of the signal.")

    # Step 1: Compute PSD using FFT
    freq_domain = np.fft.fft(signal, n=2 * len(signal))
    psd = np.abs(freq_domain) ** 2

    # Step 2: Inverse FFT to estimate autocorrelation
    autocorr = np.fft.ifft(psd).real
    autocorr = autocorr[: len(signal)]  # Use only the first half

    # Only use the first 'order + 1' elements of autocorr
    R = autocorr[: order + 1]

    # Step 3: Compute LPC coefficients using Levinson-Durbin
    # The toeplitz matrix and r vector should now have compatible dimensions
    r = R[1:]
    lpc_coeffs = solve_toeplitz((R[:-1], R[:-1]), r)

    return lpc_coeffs


def lpc_single_epoch_frame_channel(epoch_channel_data, lpc_order):
    """
    Compute LPC for a single epoch-frame-channel combination.
    """
    try:
        lpcs = fft_based_lpc(epoch_channel_data, lpc_order)
        return lpcs
    except:
        return np.zeros(lpc_order)


def process_batch(batch, lpc_order):
    # This function processes a batch of data
    batch_results = []
    for item in batch:
        result = lpc_single_epoch_frame_channel(item, lpc_order)
        batch_results.append(result)
    return batch_results


def lpc_ndarray_parallel(input_signal, lpc_order, batch_size=20):
    n_epochs, n_frames, n_channels, _ = input_signal.shape

    # Convert input_signal to a Dask array
    dask_signal = da.from_array(input_signal, chunks=(1, n_frames, n_channels, -1))

    # Create batches
    all_indices = [
        (i, j, k)
        for i in range(n_epochs)
        for j in range(n_frames)
        for k in range(n_channels)
    ]
    batches = [
        all_indices[i : i + batch_size] for i in range(0, len(all_indices), batch_size)
    ]

    # Apply process_batch to each batch
    delayed_batches = [
        delayed(process_batch)([dask_signal[idx].compute() for idx in batch], lpc_order)
        for batch in batches
    ]

    # Compute results in parallel
    computed_batches = compute(*delayed_batches)
    computed_results = [item for sublist in computed_batches for item in sublist]

    # Reshape the results back to the original 4D format
    lpc_coeffs = np.array(computed_results).reshape(
        n_epochs, n_frames, n_channels, lpc_order
    )

    return lpc_coeffs


class LPCFeatureExtractor:
    def __init__(self, n_coef: int = 10):
        self.n_coef = n_coef

    def feat_names(self, n_channels: int):
        return [
            f"LPC{coef}_ch{ch}"
            for ch in range(n_channels)
            for coef in range(self.n_coef)
        ]

    def get_feats(self, X: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"]):
        lpcs = lpc_ndarray_parallel(X, lpc_order=self.n_coef)
        return lpcs
