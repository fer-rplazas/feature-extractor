import numpy as np
from jaxtyping import Float
from numba import njit, prange

import warnings


@njit()
def causal_bl_correct(
    X: Float[np.ndarray, "n_epochs n_frames n_win_len n_features"],
    winsor_limit: float = 0.02,
    len_scale: int = 100,
):
    # Iterate over each feature
    for feat_idx in prange(X.shape[-1]):
        for epoch in prange(X.shape[0]):
            for window_length in prange(X.shape[2]):
                # Extract the data for current feature, epoch, and window length
                data = X[epoch, :, window_length, feat_idx]

                # Step 1: Winsorize the feature data
                winsorized_data = np.clip(
                    data,
                    np.quantile(data, winsor_limit),
                    np.quantile(data, 1 - winsor_limit),
                )

                # Step 2: Robustly scale each nth value using previous n-k values
                scaled_data = np.copy(winsorized_data)
                for n in range(10, len(scaled_data)):
                    id_low = max(0, n - len_scale)
                    scale_window = winsorized_data[id_low:n]
                    median = np.median(scale_window)
                    q_low = np.quantile(scale_window, 0.2)
                    q_high = np.quantile(scale_window, 0.8)
                    scale_iqr = q_high - q_low
                    # scale_iqr = iqr(scale_window)

                    if scale_iqr > 0:
                        scaled_data[n] = (scaled_data[n] - median) / (scale_iqr + 1e-8)

                # Update the features array
                X[epoch, :, window_length, feat_idx] = scaled_data

    return X


def reshape_feats(feats):
    reshaped_data = feats.stack(features=["feature_name", "win_size"])

    # Rename features to include window size
    # Drop relevant dimensions:
    # Ignore FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        features = reshaped_data.features.values
        reshaped_data.drop_vars({"feature_name", "win_size", "features"})

        reshaped_data = reshaped_data.assign_coords(
            {
                "features": [
                    f"{feat_name}_{win_size}" for feat_name, win_size in features
                ]
            }
        )
        reshaped_data = reshaped_data.stack(observation=["epoch", "frame"]).transpose()

    return reshaped_data


@njit(parallel=True)
def create_trailing_frames(
    time_series: Float[np.ndarray, "n_trials n_channels n_samples"],
    frame_length: int,
    hop_length: int,
    initial_offset: int | None = None,
) -> Float[np.ndarray, "n_trials n_frames n_channels frame_length"]:
    n_trials, n_channels, n_samples = time_series.shape

    if initial_offset is None:
        initial_offset = frame_length

    assert frame_length > 0, "Frame length should be a positive integer"
    assert (
        frame_length <= n_samples
    ), "Frame length should be less than or equal to the length of the time series"
    assert hop_length > 0, "Offset should be a positive integer"
    assert (
        initial_offset >= frame_length
    ), "Initial offset should be greater than or equal to the frame length"
    assert (
        initial_offset <= n_samples - hop_length
    ), "Initial offset should be less than the length of the time series"
    assert (
        time_series.ndim == 3
    ), "Time series should be a 3D array of shape `n_trials x n_channels x n_samples`"

    # Calculate the total number of frames
    num_frames = (n_samples - initial_offset) // hop_length + 1

    # Initialize an empty array to hold the frames
    frames = np.empty((n_trials, num_frames, n_channels, frame_length))

    # Loop over the frames
    for kk in prange(n_trials):
        for jj in prange(num_frames):
            # Calculate the start of the current frame
            start = initial_offset + jj * hop_length
            # Add the current frame to the frames array
            frames[kk, jj] = time_series[kk, :, start - frame_length : start]

    return frames


def array_idx(arr, vals):
    idx = []
    for val in vals:
        idx.append(np.argmin(np.abs(arr - val)))

    return tuple(idx)
