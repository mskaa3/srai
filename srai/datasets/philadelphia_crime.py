"""
Philadelphia Crime dataset loader.

This module contains Philadelphia Crime Dataset.
"""

from typing import Optional, Union

import geopandas as gpd

from srai.constants import WGS84_CRS
from srai.datasets import PointDataset

years_previous: list[int] = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
years_current: list[int] = [2020, 2021, 2022, 2023]


class PhiladelphiaCrimeDataset(PointDataset):
    """
    Philadelphia Crime dataset.

    Crime incidents from the Philadelphia Police Department. Part I crimes include violent offenses
    such as aggravated assault, rape, arson, among others. Part II crimes include simple assault,
    prostitution, gambling, fraud, and other non-violent offenses.
    """

    def __init__(self) -> None:
        """Create the dataset."""
        numerical_columns = None
        categorical_columns = [
            "hour",
            "dispatch_date",
            "dispatch_time",
            "dc_dist",
            "psa",
        ]
        type = "point"
        # target = "text_general_code"
        target = "count"
        super().__init__(
            "kraina/philadelphia_crime",
            type=type,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            target=target,
        )

    def _preprocessing(
        self, data: gpd.GeoDataFrame, version: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the dataset from HuggingFace.

        Args:
            data (pd.DataFrame): a dataset to preprocess
            version (str, optional): version of dataset

        Returns:
            gpd.GeoDataFrame: preprocessed data.
        """
        df = data.copy()
        gdf = gpd.GeoDataFrame(
            df.drop(["lng", "lat"], axis=1),
            geometry=gpd.points_from_xy(df["lng"], df["lat"]),
            crs=WGS84_CRS,
        )
        # TODO: Add numerical and categorical columns
        # if version in years_previous:
        #     self.numerical_columns = None
        #     self.categorical_columns = None
        # else:
        #     self.numerical_columns = None
        #     self.categorical_columns = None

        return gdf

    def load(
        self, version: Optional[Union[int, str]] = 8, hf_token: Optional[str] = None
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Method to load dataset.

        Args:
            hf_token (str, optional): If needed, a User Access Token needed to authenticate to
                the Hugging Face Hub. Environment variable `HF_TOKEN` can be also used.
                Defaults to None.
            version (str or int, optional): version of a dataset.
                Available: Official spatial train-test split from year 2023 in chosen h3 resolution:
                '8', '9, '10'. Defaults to '8'. Raw data from other years available
                as: '2013', '2014', '2015', '2016', '2017', '2018','2019', '2020', '2021',
                '2022', '2023'.

        Returns:
            dict[str, gpd.GeoDataFrame]: Dictionary with all splits loaded from the dataset. Will
                contain keys "train" and "test" if available.
        """
        return super().load(hf_token=hf_token, version=version)
