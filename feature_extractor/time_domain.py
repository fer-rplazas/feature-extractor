import numpy as np
from jaxtyping import Float
from scipy import stats


def hjorth(
    input_signal: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"],
    eps: float = 1e-7,
) -> Float[np.ndarray, "n_epochs n_frames n_channels 3"]:
    """
    Compute Hjorth activity, mobility and complexity of a time series from multichannel input

    Parameters:
    input_signal: Time series data (n_epochs x n_frames x n_channels x n_samples)

    Returns:
    Activity, Mobility, Complexity: arrays of size (n_epochs x n_frames x n_channels x 3)
    """

    # Ensure numpy array
    input_signal = np.array(input_signal)

    # Calculate the first and second derivatives
    first_deriv = np.diff(input_signal, axis=-1)
    second_deriv = np.diff(first_deriv, axis=-1)

    # Calculate variances
    var_zero = np.mean(input_signal**2, axis=-1)
    var_d1 = np.mean(first_deriv**2, axis=-1)
    var_d2 = np.mean(second_deriv**2, axis=-1)

    # Calculate Hjorth parameters
    activity = var_zero
    mobility = np.sqrt(var_d1 / (var_zero + eps))
    complexity = np.sqrt(var_d2 / (var_d1 + eps)) / (mobility + eps)

    return np.stack((activity, mobility, complexity), axis=-1)

class TimeDomainFeatureExtractor:
    def __init__(self):
        pass

    def feat_names(self, n_channels: int):
        return [
            f"{feat_name}_ch{ch}"
            for ch in range(n_channels)
            for feat_name in ("mean", "std", "kurt", "act", "mob", "complex")
        ]

    def get_feats(self, X: Float[np.ndarray, "n_epochs n_frames n_chan n_samples"]):
        mean = np.mean(X, keepdims=True, axis=-1)
        std = np.std(X, keepdims=True, axis=-1)
        kurtosis = stats.kurtosis(X, axis=-1)[..., np.newaxis]

        hjorth_feats = hjorth(X)

        return np.concatenate([mean, std, kurtosis, hjorth_feats], axis=-1)
