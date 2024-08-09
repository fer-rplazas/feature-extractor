from jaxtyping import Float
import numpy as np

from joblib import Parallel, delayed
import numpy as np
from statsmodels.regression.linear_model import yule_walker


def compute_ar_params(signal, order):
    rho, sigma = yule_walker(signal, order=order)
    return rho


def process_channel(channel_data, ar_order):
    return np.apply_along_axis(
        compute_ar_params, axis=-1, arr=channel_data, order=ar_order
    )


class ARFeatureExtractor:
    def __init__(self, order: int = 12):
        self.order = order

    def feat_names(self, n_channels: int):
        return [
            f"AR{coef}_ch{ch}" for ch in range(n_channels) for coef in range(self.order)
        ]

    def get_feats(
        self, X: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"]
    ) -> Float[np.ndarray, "n_epochs n_frames n_channels n_coef"]:

        window = np.hanning(X.shape[-1])
        X = X * window

        n_epochs = X.shape[0]

        # Use joblib to process each channel in parallel
        ar_params = Parallel(n_jobs=-1)(
            delayed(process_channel)(X[i], self.order) for i in range(X.shape[0])
        )

        ar_params = np.stack(ar_params, axis=0)

        return ar_params
