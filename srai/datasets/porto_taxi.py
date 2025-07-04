"""
Porto Taxi dataset loader.

This module contains Porto Taxi Dataset.
"""

from typing import Optional, Union

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from tqdm import tqdm

from srai.constants import WGS84_CRS
from srai.datasets import TrajectoryDataset


class PortoTaxiDataset(TrajectoryDataset):
    """
    Porto Taxi dataset.

    The dataset covers a year of trajectory data for taxis in Porto, Portugal
    Each ride is categorized as:
    A) taxi central based,
    B) stand-based or
    C) non-taxi central based.
    Each data point represents a completed trip initiated through
    the dispatch central, a taxi stand, or a random street.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = ["speed"]
        categorical_columns = ["call_type", "origin_call", "origin_stand", "day_type"]
        type = "trajectory"
        target = "trip_id"
        # target = None
        super().__init__(
            "kraina/porto_taxi",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _aggregate_trajectories_to_hexes(
        self,
        gdf: gpd.GeoDataFrame,
        resolution: int,
        version: str,
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the gdf with linestring trajectories to h3 trajectories.

        Args:
            gdf (gpd.DataFrame): a gdf with prepared linestring.
            resolution (int) : h3 resolution to regionalize data.
            version (str): version of dataset.

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        tqdm.pandas(desc="Building h3 trajectories")
        _gdf = gdf.copy()
        # if version == "TTE":

        def process_row(row: pd.Series) -> pd.Series:
            """
            Process a single row containing a trajectory LineString and associated metadata.

            Converting to H3 hexagons.

            Args:
                row (pd.Series): A row from a GeoDataFrame with at least the following fields:
                    - geometry (shapely.geometry.LineString)
                    - speed (list[float])
                    - timestamp (list[pd.Timestamp])
                    - day_type (str)
                    - call_type (str)
                    - taxi_id (int or str)

            Returns:
                pd.Series: A new series containing:
                    - duration (float): Duration of the trajectory in seconds.
                    - h3_sequence (list[str]): List of H3 hex IDs visited.
                    - avg_speed_per_hex (list[float]): Average speed per H3 hex.
                    - day_type, call_type, taxi_id, geometry: Carried over from input.
            """
            coords = row.geometry.coords
            speeds = row["speed"]
            timestamps = row["timestamp"]

            duration = (timestamps[-1] - timestamps[0]).total_seconds()

            raw_h3_seq = []
            hex_metadata = []

            for (lon, lat), speed, ts in zip(coords, speeds, timestamps):
                hex_id = h3.latlng_to_cell(lat, lon, resolution)
                raw_h3_seq.append(hex_id)
                hex_metadata.append(
                    {
                        "hex_id": hex_id,
                        "speed": speed,
                        "timestamp": ts,
                    }
                )

            # Remove consecutive duplicates
            cleaned_seq = [hex_metadata[0]]
            for meta in hex_metadata[1:]:
                if meta["hex_id"] != cleaned_seq[-1]["hex_id"]:
                    cleaned_seq.append(meta)

            # Interpolate missing hexes and propagate metadata
            full_seq: list[str] = []
            speeds_interp: list[float] = []
            timestamps_interp: list[pd.Timestamp] = []

            for i in range(len(cleaned_seq) - 1):
                start = cleaned_seq[i]
                end = cleaned_seq[i + 1]
                last_known_speed = start["speed"]
                last_known_ts = start["timestamp"]

                try:
                    path = h3.grid_path_cells(start["hex_id"], end["hex_id"])
                    if path:
                        # Skip first to avoid duplication
                        if full_seq and path[0] == full_seq[-1]:
                            path = path[1:]
                        for h in path:
                            full_seq.append(h)
                            speeds_interp.append(last_known_speed)
                            timestamps_interp.append(last_known_ts)
                except Exception:
                    full_seq.append(start["hex_id"])
                    speeds_interp.append(last_known_speed)
                    timestamps_interp.append(last_known_ts)

            # Add last
            last = cleaned_seq[-1]
            full_seq.append(last["hex_id"])
            speeds_interp.append(last["speed"])
            timestamps_interp.append(last["timestamp"])

            res = {
                "trip_id": row["trip_id"],
                "h3_sequence": full_seq,
                "avg_speed_per_hex": speeds_interp,
                "timestamp": timestamps_interp,
                "day_type": row["day_type"],
                "call_type": row["call_type"],
                "taxi_id": row["taxi_id"],
                "geometry": row.geometry,
            }

            if version == "TTE":
                res["duration"] = duration
            elif version == "HMP":
                split_idx = int(len(full_seq) * 0.85)
                if split_idx == len(full_seq):
                    split_idx = len(full_seq) - 1
                del res["h3_sequence"]
                res["h3_sequence_x"] = full_seq[:split_idx]
                res["h3_sequence_y"] = full_seq[split_idx:]

            return pd.Series(res)

        # Apply with progress bar
        hexes_df = _gdf.progress_apply(process_row, axis=1)
        if version == "HMP":
            hexes_df = hexes_df[
                hexes_df["h3_sequence_x"].apply(lambda x: len(x) > 0)
                & hexes_df["h3_sequence_y"].apply(lambda y: len(y) > 0)
            ].reset_index(drop=True)
        elif version == "TTE":
            hexes_df = hexes_df[hexes_df["h3_sequence"].apply(lambda x: len(x) > 0)].reset_index(
                drop=True
            )
            hexes_df = hexes_df[hexes_df["duration"] > 0.0].reset_index(drop=True)
        hexes_gdf = gpd.GeoDataFrame(hexes_df, geometry="geometry", crs=WGS84_CRS)

        return hexes_gdf

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data: Data of Chicago Crime dataset.
            version: version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # <- fix here

        gdf = gpd.GeoDataFrame(
            df.drop(["longitude", "latitude"], axis=1),
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=WGS84_CRS,
        )

        assert self.target is not None
        assert self.resolution is not None
        assert self.version is not None
        trajectory_gdf = self._agg_points_to_trajectories(gdf=gdf, target_column=self.target)

        hexes_gdf = self._aggregate_trajectories_to_hexes(
            gdf=trajectory_gdf, resolution=self.resolution, version=self.version
        )
        lengths = hexes_gdf.geometry.length

        # Compute 5th and 95th percentiles
        lower = np.percentile(lengths, 5)
        upper = np.percentile(lengths, 95)

        # Filter based on length
        hexes_gdf = hexes_gdf[(lengths >= lower) & (lengths <= upper)]
        return hexes_gdf

    def load(
        self,
        version: Optional[Union[int, str]] = "TTE",
        hf_token: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (Optional[str]): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (Optional[str, int]): version of a dataset.
                Available: Official train-test split for Travel Time Estimation task (TTE) and
                Human Mobility Prediction task (HMP). Raw data from available as: 'all'.
            resolution (Optional[int]): H3 resolution for hex trajectories.
                Neccessary if using 'all' split.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        if version == "TTE" or version == "HMP":
            self.resolution = 9
        elif version == "all":
            if resolution is None:
                raise ValueError("Pass the resolution parameter to generate h3 trajectories.")
            else:
                self.resolution = resolution
        else:
            raise NotImplementedError("Version not implemented")
        return super().load(hf_token=hf_token, version=version)
