# Description: This file contains the modalities that you wish to use for training
# Provide all the modalities that you wish to use for training, and also the corresponding bands
# NOTE: THE MAIN DICT TO CHANGE IS THE INP_MODALITIES AND OUT_MODALITIES. 
import os
from pathlib import Path

# we have the data for both l2a and l1c, so l2a has everything below except B10. l1c has everything except SCL and MSK_CLDPRB
# sentinel2: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
# sentinel2_cloudmask: QA60
# sentinel2_cloudprod: MSK_CLDPRB
# sentinel2_scl: SCL
# sentinel1: asc_VV, asc_VH, asc_HH, asc_HV, desc_VV, desc_VH, desc_HH, desc_HV
# aster: elevation, slope
# era5:
# prev_month: avg_temp, min_temp, max_temp, total_precip
# curr_month: avg_temp, min_temp, max_temp, total_precip
# year: avg_temp, min_temp, max_temp, total_precip
# dynamic_world: landcover
# canopy_height_eth: height, std
# lat
# lon
# biome
# eco_region
# month
# esa_worldcover: map


# provide the bands and modalities based on the names above. if you just want all bands, just mention 'all' with the corresponding modalities.

_MMEARTH_DIR_ENV = os.environ.get("MMEARTH_DIR", None)

MMEARTH_DIR = Path("/projects/dereeco/data/global-lr/data_1M_130_new/")
if _MMEARTH_DIR_ENV is not None:
    MMEARTH_DIR = Path(_MMEARTH_DIR_ENV)

NO_DATA_VAL = {
    "sentinel2": 0,
    "sentinel2_cloudmask": 65535,
    "sentinel2_cloudprod": 65535,
    "sentinel2_scl": 255,
    "sentinel1": float("-inf"),
    "aster": float("-inf"),
    "canopy_height_eth": 255,
    "dynamic_world": 0,
    "esa_worldcover": 255,
    "lat": float("-inf"),
    "lon": float("-inf"),
    "month": float("-inf"),
    "era5": float("inf"),
    "biome": 255,
    "eco_region": 65535,
}

# Input modalities for training
INP_MODALITIES = {
    "sentinel2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B8",
        "B9",
        "B11",
        "B12",
    ],
}


# Output modalities for training
OUT_MODALITIES = {
    "sentinel2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B8",
        "B9",
        "B11",
        "B12",
    ],
    "sentinel1": "all",
    "aster": "all",
    "era5": "all",
    "dynamic_world": "all",
    "canopy_height_eth": "all",
    "lat": "all",
    "lon": "all",
    "biome": "all",
    "eco_region": "all",
    "month": "all",
    "esa_worldcover": "all",
}


RGB_MODALITIES = {
    "sentinel2": ["B2", "B3", "B4"],
}

# an example of all the modalities. DO NOT CHANGE THIS, ALWAYS CHANGE THE INP and OUT MODALITIES ABOVE
MODALITIES_FULL = {
    "sentinel2": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
    ],
    "sentinel2_cloudmask": ["QA60"],
    "sentinel2_cloudprod": ["MSK_CLDPRB"],
    "sentinel2_scl": ["SCL"],
    "sentinel1": [
        "asc_VV",
        "asc_VH",
        "asc_HH",
        "asc_HV",
        "desc_VV",
        "desc_VH",
        "desc_HH",
        "desc_HV",
    ],
    "aster": ["elevation", "slope"],
    "era5": [
        "prev_month_avg_temp",
        "prev_month_min_temp",
        "prev_month_max_temp",
        "prev_month_total_precip",
        "curr_month_avg_temp",
        "curr_month_min_temp",
        "curr_month_max_temp",
        "curr_month_total_precip",
        "year_avg_temp",
        "year_min_temp",
        "year_max_temp",
        "year_total_precip",
    ],
    "dynamic_world": ["landcover"],
    "canopy_height_eth": ["height", "std"],
    "lat": ["sin", "cos"],
    "lon": ["sin", "cos"],
    "biome": ["biome"],
    "eco_region": ["eco_region"],
    "month": ["sin_month", "cos_month"],
    "esa_worldcover": ["map"],
}

MODALITY_TASK = {
    # map regression
    "sentinel2": "regression_map",
    "sentinel1": "regression_map",
    "aster": "regression_map",
    "canopy_height_eth": "regression_map",
    # pixel regression
    "lat": "regression",
    "lon": "regression",
    "month": "regression",
    "era5": "regression",
    # semantic segmentation
    "esa_worldcover": "segmentation",
    "dynamic_world": "segmentation",
    # pixel classification
    "biome": "classification",
    "eco_region": "classification",
}

PIXEL_WISE_MODALITIES = [
    "sentinel2",
    "sentinel1",
    "aster",
    "canopy_height_eth",
    "esa_worldcover",
    "dynamic_world",
]
