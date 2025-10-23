# %%
import numpy as np

from feature_extractor import FeatureExtractor

# %%
fs = 4096.0
signals = np.random.randn(32, 16, 4096 * 8)

fe = FeatureExtractor(
    fs=fs, n_jobs=96, device="cpu", modes=["periodogram", "periodogram_norm", "time"]
)
fe.calculate_features(signals)
# %%
fe.features

# %%
