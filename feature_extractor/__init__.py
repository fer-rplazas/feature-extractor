from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Float

from .cepstral import CepstralFeatureExtractor
from .coherence import CoherenceFeatureExtractor
from .lpc import LPCFeatureExtractor
from .ar import ARFeatureExtractor
from .periodogram import PeriodogramFeatureExtractor
from .plv import PLVFeatureExtractor
from .psi import PSIFeatureExtractor
from .time_domain import TimeDomainFeatureExtractor
from .utils import (
    causal_bl_correct,
    create_trailing_frames,
    ensure_device_array,
    to_numpy,
)


def _resolve_device(device: str | jax.Device | None) -> jax.Device | None:
    if device is None:
        return None
    if isinstance(device, jax.Device):
        return device
    if isinstance(device, str):
        try:
            devices = jax.devices(device)
        except RuntimeError as err:
            raise ValueError(f"No JAX devices available for platform '{device}'.") from err
        if not devices:
            raise ValueError(f"No JAX devices available for platform '{device}'.")
        return devices[0]
    raise TypeError("device must be a string, jax.Device, or None.")


class FeatureExtractor:
    def __init__(
        self,
        fs: float,
        win_sizes: list[int] | None = None,
        hop_len: int | None = None,
        initial_offset: int | None = None,
        modes: list[str] | None = None,
        coh_channels: list[tuple[int, int]] | None = None,
        n_jobs: int | None = None,
        periodogram_kwargs: dict | None = None,
        time_kwargs: dict | None = None,
        cepstrum_kwargs: dict | None = None,
        ar_kwargs: dict | None = None,
        coherence_kwargs: dict | None = None,
        plv_kwargs: dict | None = None,
        psi_kwargs: dict | None = None,
        device: str | jax.Device | None = None,
    ):
        self.fs = fs
        self.win_sizes = win_sizes or [
            int(0.5 * self.fs),
            int(1 * self.fs),
        ]
        self.hop_len = hop_len or int(0.01 * self.fs)
        self.initial_offset = initial_offset or max(self.win_sizes)
        self.modes = modes or [
            "time",
            "periodogram",
            "periodogram_norm",
            "cepstrum",
        ]
        self.coh_channels = coh_channels or [(0, 1)]
        self.n_jobs = n_jobs or 1

        self.periodogram_kwargs = periodogram_kwargs or {}
        self.time_kwargs = time_kwargs or {}
        self.cepstrum_kwargs = cepstrum_kwargs or {}
        self.coherence_kwargs = coherence_kwargs or {}
        self.ar_kwargs = ar_kwargs or {}
        self.plv_kwargs = plv_kwargs or {}
        self.psi_kwargs = psi_kwargs or {}

        self.device = _resolve_device(device)

        self.features = None

    def calculate_features(
        self, data: Float[np.ndarray, "n_epochs n_channels n_samples"]
    ):
        dtype = data.dtype if isinstance(data, jax.Array) else np.asarray(data).dtype
        data_device = ensure_device_array(
            data,
            device=self.device,
            dtype=jax.dtypes.canonicalize_dtype(dtype),
        )

        if self.n_jobs > 1:
            mode_workers = max(1, self.n_jobs // max(1, len(self.win_sizes)))
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(
                        self.get_feats_for_win_size,
                        data_device,
                        win_size,
                        mode_workers,
                    )
                    for win_size in self.win_sizes
                ]
                feat_cont = [future.result() for future in futures]
        else:
            mode_workers = 1
            feat_cont = [
                self.get_feats_for_win_size(data_device, win_size, mode_workers)
                for win_size in self.win_sizes
            ]

        # Concatenate, assign coordinates, and transpose as before
        self.features = xr.concat(feat_cont, dim="win_size")
        self.features = self.features.assign_coords(
            win_size=np.array(self.win_sizes) / self.fs
        )
        self.features = self.features.transpose(
            "epoch", "frame", "win_size", "feature_name"
        )
        self.features.attrs["fs"] = self.fs

        return self

    def causal_bl_correct(self, winsor_limit: float = 0.02, len_time: float = 5.0):
        if not hasattr(self, "features"):
            raise ValueError(
                "Features have not been calculated yet -- call `calculate_features` first"
            )

        dt = self.features.frame[1] - self.features.frame[0]
        len_samples = int(len_time / dt)

        self.features.values = causal_bl_correct(
            self.features.values, winsor_limit, len_samples, device=self.device
        )

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
        framed: Float[jax.Array, "n_epochs n_frames n_channels n_samples"],
        t_framed: Float[np.ndarray, "n_frames"],
    ) -> list[xr.DataArray]:
        features_xr_cont = []
        framed_np: np.ndarray | None = None

        if mode == "time":
            time_kwargs = dict(self.time_kwargs)
            time_kwargs.setdefault("device", self.device)
            time_ext = TimeDomainFeatureExtractor(**time_kwargs)
            feats = time_ext.get_feats(framed)
            feats = feats.reshape(feats.shape[:-2] + (-1,))
            feats_np = to_numpy(feats)

            features_xr_cont.append(
                xr.DataArray(
                    feats_np,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats_np.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": time_ext.feat_names(framed.shape[2]),
                    },
                )
            )

        elif mode == "LPC":
            extractor = LPCFeatureExtractor()
            if framed_np is None:
                framed_np = to_numpy(framed)
            feats = extractor.get_feats(framed_np)
            feats = feats.reshape(feats.shape[:-2] + (-1,))
            feats_np = to_numpy(feats)

            features_xr_cont.append(
                xr.DataArray(
                    feats_np,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats_np.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": extractor.feat_names(framed.shape[2]),
                    },
                )
            )

        elif mode == "ar":
            ar_ext = ARFeatureExtractor(**self.ar_kwargs)
            if framed_np is None:
                framed_np = to_numpy(framed)
            feats = ar_ext.get_feats(framed_np)
            feats = feats.reshape(feats.shape[:-2] + (-1,))
            feats_np = to_numpy(feats)

            features_xr_cont.append(
                xr.DataArray(
                    feats_np,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats_np.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": ar_ext.feat_names(framed.shape[2]),
                    },
                )
            )

        elif mode == "cepstrum":
            cepstrum_kwargs = dict(self.cepstrum_kwargs)
            cepstrum_kwargs.setdefault("device", self.device)
            cepstrum_ext = CepstralFeatureExtractor(**cepstrum_kwargs)
            feats = cepstrum_ext.get_feats(framed)
            feats = feats.reshape(feats.shape[:-2] + (-1,))
            feats_np = to_numpy(feats)

            features_xr_cont.append(
                xr.DataArray(
                    feats_np,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats_np.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": cepstrum_ext.feat_names(framed.shape[2]),
                    },
                )
            )

        elif mode == "periodogram":
            periodogram_kwargs = dict(self.periodogram_kwargs)
            periodogram_kwargs.setdefault("device", self.device)
            periodogram_ext = PeriodogramFeatureExtractor(**periodogram_kwargs)
            feats = periodogram_ext.get_feats(framed, fs=self.fs)
            feats = feats.reshape(feats.shape[:-2] + (-1,))
            feats_np = to_numpy(feats)

            features_xr_cont.append(
                xr.DataArray(
                    feats_np,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats_np.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": periodogram_ext.feat_names(framed.shape[2]),
                    },
                )
            )

        elif mode == "periodogram_norm":
            periodogram_kwargs = dict(self.periodogram_kwargs)
            periodogram_kwargs.setdefault("device", self.device)
            periodogram_ext = PeriodogramFeatureExtractor(
                **periodogram_kwargs, normalize=True
            )
            feats = periodogram_ext.get_feats(framed, fs=self.fs)
            feats = feats.reshape(feats.shape[:-2] + (-1,))
            feats_np = to_numpy(feats)

            features_xr_cont.append(
                xr.DataArray(
                    feats_np,
                    dims=["epoch", "frame", "feature_name"],
                    coords={
                        "epoch": np.arange(feats_np.shape[0]),
                        "frame": t_framed.reshape(-1),
                        "feature_name": periodogram_ext.feat_names(framed.shape[2]),
                    },
                )
            )

        # Connectivity-based features
        elif mode == "coh":
            if framed_np is None:
                framed_np = to_numpy(framed)
            for ch1, ch2 in self.coh_channels:
                coherence_ext = CoherenceFeatureExtractor(**self.coherence_kwargs)
                feats = coherence_ext.get_feats(
                    framed_np[:, :, ch1, :],
                    framed_np[:, :, ch2, :],
                    fs=self.fs,
                )
                feats_np = to_numpy(feats)

                features_xr_cont.append(
                    xr.DataArray(
                        feats_np,
                        dims=["epoch", "frame", "feature_name"],
                        coords={
                            "epoch": np.arange(feats_np.shape[0]),
                            "frame": t_framed.reshape(-1),
                            "feature_name": [
                                f"{name}_ch{ch1}-{ch2}"
                                for name in coherence_ext.feat_names()
                            ],
                        },
                    )
                )

        elif mode == "psi":
            if framed_np is None:
                framed_np = to_numpy(framed)
            for ch1, ch2 in self.coh_channels:
                coherence_ext = PSIFeatureExtractor(**self.psi_kwargs)
                feats = coherence_ext.get_feats(
                    framed_np[:, :, ch1, :],
                    framed_np[:, :, ch2, :],
                    fs=self.fs,
                )
                feats_np = to_numpy(feats)

                features_xr_cont.append(
                    xr.DataArray(
                        feats_np,
                        dims=["epoch", "frame", "feature_name"],
                        coords={
                            "epoch": np.arange(feats_np.shape[0]),
                            "frame": t_framed.reshape(-1),
                            "feature_name": [
                                f"PSI{freqs}_ch{ch1}-{ch2}"
                                for freqs in coherence_ext.feat_names()
                            ],
                        },
                    )
                )

        elif mode == "plv":
            if framed_np is None:
                framed_np = to_numpy(framed)
            for ch1, ch2 in self.coh_channels:
                coherence_ext = PLVFeatureExtractor(**self.plv_kwargs)
                feats = coherence_ext.get_feats(
                    framed_np[:, :, ch1, :],
                    framed_np[:, :, ch2, :],
                    fs=self.fs,
                )
                feats_np = to_numpy(feats)

                features_xr_cont.append(
                    xr.DataArray(
                        feats_np,
                        dims=["epoch", "frame", "feature_name"],
                        coords={
                            "epoch": np.arange(feats_np.shape[0]),
                            "frame": t_framed.reshape(-1),
                            "feature_name": [
                                f"PLV{freqs}_ch{ch1}-{ch2}"
                                for freqs in coherence_ext.feat_names()
                            ],
                        },
                    )
                )
        else:
            raise ValueError(f"Unknown feat class modality '{mode}'")

        return features_xr_cont

    def get_feats_for_win_size(
        self,
        data: jax.Array,
        win_size: int,
        mode_workers: int = 1,
    ):
        framed = create_trailing_frames(
            data,
            win_size,
            self.hop_len,
            self.initial_offset,
            device=self.device,
        )
        t = jnp.arange(data.shape[-1], dtype=jnp.float32) / self.fs
        t = t.reshape(1, 1, -1)
        t_framed = create_trailing_frames(
            t,
            win_size,
            self.hop_len,
            self.initial_offset,
            device=self.device,
        )
        t_framed = jnp.squeeze(t_framed[..., -1])
        t_framed_np = to_numpy(t_framed)

        if mode_workers > 1 and len(self.modes) > 1:
            max_workers = min(mode_workers, len(self.modes))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._process_mode, mode, framed, t_framed_np)
                    for mode in self.modes
                ]
                features_dicts = []
                for future in futures:
                    features_dicts.extend(future.result())
        else:
            features_dicts = []
            for mode in self.modes:
                features_dicts.extend(self._process_mode(mode, framed, t_framed_np))

        return xr.concat(features_dicts, dim="feature_name")
