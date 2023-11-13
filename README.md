# Installation

## Install dependencies
- dask
- numba
- xarray
- jaxtyping


With conda (recommended):

```
conda install -c conda-forge dask numba xarray
pip install jaxtyping
```

## Install package into environment in editable/development mode
```
pip install -e /path/to/this/folder
```


# Usage
Import into your scripts

```python
from feature_extractor import FeatureExtractor


extractor = FeatureExtractor(n_jobs=-1, fs = 4096)
extractor = calculate_features(X) # X is an array of shape (n_epochs x n_channels x n_samples)
features = extractor.features
```

