"""
AirbnbMulticity dataset loader.

This module contains AirbnbMulticity dataset.
"""

from typing import Optional

import geopandas as gpd
import pandas as pd

from srai.constants import WGS84_CRS
from srai.datasets import HuggingFaceDataset


class AirbnbMulticityDataset(HuggingFaceDataset):
    """
    AirbnbMulticity dataset.

    Dataset description will be added.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        categorical_columns = ["name", "host_name", "neighborhood", "room_type", "city"]
        numerical_columns = [
            "number_of_reviews",
            "minimum_nights",
            "availability_365",
            "calculated_host_listings_count",
            "number_of_reviews_ltm",
        ]
        target = "price"
        type = "point"
        super().__init__(
            "kraina/airbnb_multicity",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(self, data: pd.DataFrame, version: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Preprocessing to get GeoDataFrame with location data, based on GEO_EDA files.

        Args:
            data (pd.DataFrame): Data of AirbnbMulticity dataset.
            version (str, optional): version of a dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["latitude", "longitude"], axis=1),
            geometry=gpd.points_from_xy(x=df["longitude"], y=df["latitude"]),
            crs=WGS84_CRS,
        )

        return gdf

    def load(
        self, hf_token: Optional[str] = None, version: Optional[str] = "8"
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str, optional): version of a dataset.
                Available: '8', '9', '10', where number is a h3 resolution used in train-test \
                    split. Benchmark version comprises six cities: Paris, Rome, London, Amsterdam, \
                        Melbourne, New York City. Raw, full data from ~80 cities available as 'all'.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        return super().load(hf_token, version)
