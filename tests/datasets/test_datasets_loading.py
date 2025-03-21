"""Tests for dataset loading."""

from typing import Any, Optional

import pytest

from srai.datasets import (
    AirbnbMulticityDataset,
    ChicagoCrimeDataset,
    HouseSalesInKingCountyDataset,
    PhiladelphiaCrimeDataset,
    PoliceDepartmentIncidentsDataset,
)


@pytest.mark.parametrize(  # type: ignore
    "dataset_class,version,train_size,test_size",
    [
        (AirbnbMulticityDataset, "8", 605272, 151319),
        (ChicagoCrimeDataset, "9", 188381, 46536),
        (HouseSalesInKingCountyDataset, "8", 17290, 4323),
        (PhiladelphiaCrimeDataset, "8", 128765, 33187),
        (PoliceDepartmentIncidentsDataset, "9", 621263, 164811),
        (AirbnbMulticityDataset, "all", 3099825, None),
        (ChicagoCrimeDataset, "2022", 234919, None),
        (HouseSalesInKingCountyDataset, "all", 21613, None),
        (PhiladelphiaCrimeDataset, "2013", 141352, None),
        (PoliceDepartmentIncidentsDataset, "all", 786074, None),
    ],
)
def test_dataset_loading(
    dataset_class: Any,
    version: str,
    train_size: int,
    test_size: Optional[int],
) -> None:
    """Test if a dataset can be loaded and assert correct length."""
    dataset = dataset_class()
    dataset.load(version=version)

    assert len(dataset.train_gdf) == train_size
    if test_size:
        assert len(dataset.test_gdf) == test_size
    else:
        assert dataset.test_gdf is None
