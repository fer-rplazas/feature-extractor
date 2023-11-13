import numpy as np
import xarray as xr
from dask import compute, delayed
from jaxtyping import Float

from .utils import create_trailing_frames
from .time_domain import TimeDomainFeatureExtractor
from .cepstral import CepstralFeatureExtractor
from .periodogram import PeriodogramFeatureExtractor
from .lpc import LPCFeatureExtractor
from .coherence import CoherenceFeatureExtractor
from .plv import PLVFeatureExtractor


class FeatureExtractor:
    def __init__(
        self,
        win_sizes: list[int] | None = None,
        hop_len: int | None = None,
        initial_offset: int | None = None,
        modes: list[str] | None = None,
        coh_channels: list[tuple[int, int]] | None = None,
        n_jobs: int | None = None,
        fs: float = 2048.0,
    ):
        self.fs = fs
        self.win_sizes = win_sizes or [
            int(0.5 * self.fs),
            int(1 * self.fs),
            int(2 * self.fs),
        ]
        self.hop_len = hop_len or int(0.01 * self.fs)
        self.initial_offset = initial_offset or max(self.win_sizes)
        self.modes = modes or ["time", "periodogram", "cepstrum"]
        self.coh_channels = coh_channels or [(0, 1)]
        self.n_jobs = n_jobs or 1
        self.features = None

    def calculate_features(
        self, data: Float[np.ndarray, "n_epochs n_channels n_samples"]
    ):
        if True:  # Cannot use dask here due to memory shortage
            feat_cont = [
                self.get_feats_for_win_size(data, win_size)
                for win_size in self.win_sizes
            ]
        else:
            # Create delayed tasks
            jobs = [
                delayed(self.get_feats_for_win_size)(data, win_size)
                for win_size in self.win_sizes
            ]

            # Run jobs in parallel using Dask
            feat_cont = compute(*jobs)

        # Concatenate, assign coordinates, and transpose as before
        self.features = xr.concat(feat_cont, dim="win_size")
        self.features = self.features.assign_coords(win_size=self.win_sizes)
        self.features = self.features.transpose(
            "epoch", "frame", "win_size", "feature_name"
        )

        return self

    def label_from_frame_time(self, t_cutoff: float, n_epochs: int):
        if self.features is None:
            raise ValueError(
                "Features have not been calculated yet -- call `calculate_features` first"
            )

        # Single frame vector:
        frame_label = self.features.frame > t_cutoff

        # Whole feat_mat:
        feat_mat_label = np.tile(frame_label, (n_epochs, 1))

        feat_mat_label = feat_mat_label.reshape(1, -1)

        return frame_label, feat_mat_label

    def feat_mat(
        self,
        select_epochs: None | list[int] = None,
        exclude_epochs: None | list[int] = None,
    ):
        if self.features is None:
            raise ValueError(
                "Features have not been calculated yet -- call `calculate_features` first"
            )

        if select_epochs is not None and exclude_epochs is not None:
            raise ValueError("Cannot select and exclude epochs at the same time")

        if select_epochs is not None:
            feats = self.features.sel(epoch=np.array(select_epochs)).to_numpy()

        elif exclude_epochs is not None:
            all_epochs = self.features.epoch.to_numpy()
            # Get the set difference between all epochs and excluded epochs
            select_from_diff = np.setdiff1d(all_epochs, exclude_epochs)
            feats = self.features.sel(epoch=select_from_diff).to_numpy()

        else:
            # If neither select nor exclude is specified, use all epochs
            feats = self.features.to_numpy()

        return feats.reshape(
            feats.shape[0] * feats.shape[1], feats.shape[2] * feats.shape[3]
        )

    def _process_mode(
        self,
        mode: str,
        framed: Float[np.ndarray, "n_epochs n_frames n_channels n_samples"],
        t_framed: Float[np.ndarray, "n_frames"],
    ):
        if mode == "time":
            time_ext = TimeDomainFeatureExtractor()
            feats = time_ext.get_feats(framed)
            feats = feats.reshape(feats.shape[:-2] + (-1,))

            features_xr = xr.DataArray(
                feats,  # the numpy array of features
                dims=["epoch", "frame", "feature_name"],
                coords={
                    "epoch": np.arange(feats.shape[0]),
                    "frame": t_framed.reshape(-1),
                    "feature_name": time_ext.feat_names(framed.shape[2]),
                },
            )

        elif mode == "LPC":
            extractor = LPCFeatureExtractor()
            feats = extractor.get_feats(framed)
            feats = feats.reshape(feats.shape[:-2] + (-1,))

            features_xr = xr.DataArray(
                feats,  # the numpy array of features
                dims=["epoch", "frame", "feature_name"],
                coords={
                    "epoch": np.arange(feats.shape[0]),
                    "frame": t_framed.reshape(-1),
                    "feature_name": extractor.feat_names(framed.shape[2]),
                },
            )

        elif mode == "cepstrum":
            cepstrum_ext = CepstralFeatureExtractor()
            feats = cepstrum_ext.get_feats(framed)
            feats = feats.reshape(feats.shape[:-2] + (-1,))

            features_xr = xr.DataArray(
                feats,  # the numpy array of features
                dims=["epoch", "frame", "feature_name"],
                coords={
                    "epoch": np.arange(feats.shape[0]),
                    "frame": t_framed.reshape(-1),
                    "feature_name": cepstrum_ext.feat_names(framed.shape[2]),
                },
            )

        elif mode == "periodogram":
            periodogram_ext = PeriodogramFeatureExtractor()
            feats = periodogram_ext.get_feats(framed, fs=self.fs)
            feats = feats.reshape(feats.shape[:-2] + (-1,))

            features_xr = xr.DataArray(
                feats,  # the numpy array of features
                dims=["epoch", "frame", "feature_name"],
                coords={
                    "epoch": np.arange(feats.shape[0]),
                    "frame": t_framed.reshape(-1),
                    "feature_name": periodogram_ext.feat_names(framed.shape[2]),
                },
            )

        # Connectivity features
        elif mode == "coh":
            for ch1, ch2 in self.coh_channels:
                coherence_ext = CoherenceFeatureExtractor()
                feats = coherence_ext.get_feats(
                    framed[:, :, ch1, :], framed[:, :, ch2, :], fs=self.fs
                )

                features_xr = xr.DataArray(
                    feats,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": [
                            f"{name}_ch{ch1}-{ch2}"
                            for name in coherence_ext.feat_names()
                        ],
                    },
                )

        elif mode == "plv":
            for ch1, ch2 in self.coh_channels:
                coherence_ext = PLVFeatureExtractor()
                feats = coherence_ext.get_feats(
                    framed[:, :, ch1, :], framed[:, :, ch2, :], fs=self.fs
                )

                features_xr = xr.DataArray(
                    feats,  # the numpy array of features
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": [
                            f"PLV{freqs}_ch{ch1}-{ch2}"
                            for freqs in coherence_ext.feat_names()
                        ],
                    },
                )
        else:
            raise ValueError(f"Unknown feat class modality '{mode}'")

        return features_xr

    def get_feats_for_win_size(self, data, win_size):
        framed = create_trailing_frames(
            data, win_size, self.hop_len, self.initial_offset
        )
        t = np.arange(data.shape[-1]) / self.fs
        t_framed = create_trailing_frames(
            np.expand_dims(t, (0, 1)), win_size, self.hop_len, self.initial_offset
        )[:, :, :, -1]
        t_framed = np.squeeze(t_framed)

        if self.n_jobs == 1:
            # Sequential execution
            features_dicts = [
                self._process_mode(mode, framed, t_framed) for mode in self.modes
            ]
        else:
            # Parallel execution with Dask
            delayed_features = [
                delayed(self._process_mode)(mode, framed, t_framed)
                for mode in self.modes
            ]
            features_dicts = compute(*delayed_features)

        return xr.concat(features_dicts, dim="feature_name")
