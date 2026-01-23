"""
conftest.py for pytest configuration.
"""

from datetime import datetime

import pytest


@pytest.fixture
def example_correct_data() -> dict:
    """
    Example of correct ome-arrow data for testing.
    """
    return {
        "type": "ome.arrow",
        "version": "1.0.0",
        "id": "img-0001",
        "name": "Example image",
        "image_type": "image",
        "acquisition_datetime": datetime(2025, 1, 1, 12, 0, 0),
        "pixels_meta": {
            "dimension_order": "XYCT",  # Z==1, so XYCT is fine
            "type": "uint16",
            "size_x": 4,  # width
            "size_y": 3,  # height
            "size_z": 1,  # one z-slice
            "size_c": 2,  # two channels
            "size_t": 1,  # one timepoint
            "physical_size_x": 0.65,
            "physical_size_y": 0.65,
            "physical_size_z": 1.00,
            "physical_size_x_unit": "µm",
            "physical_size_y_unit": "µm",
            "physical_size_z_unit": "µm",
            "channels": [
                {
                    "id": "C0",
                    "name": "DNA",
                    "emission_um": 0.46,
                    "excitation_um": 0.40,
                    "illumination": "Epifluorescence",
                    "color_rgba": 0x0000FFFF,  # blue-ish
                },
                {
                    "id": "C1",
                    "name": "Mito",
                    "emission_um": 0.59,
                    "excitation_um": 0.54,
                    "illumination": "Epifluorescence",
                    "color_rgba": 0xFF0000FF,  # red-ish
                },
            ],
        },
        "chunk_grid": {
            "order": "TCZYX",
            "chunk_t": 1,
            "chunk_c": 1,
            "chunk_z": 1,
            "chunk_y": 3,
            "chunk_x": 4,
            "chunk_order": "ZYX",
        },
        "chunks": [
            {
                "t": 0,
                "c": 0,
                "z": 0,
                "y": 0,
                "x": 0,
                "shape_z": 1,
                "shape_y": 3,
                "shape_x": 4,
                "pixels": [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
            },
            {
                "t": 0,
                "c": 1,
                "z": 0,
                "y": 0,
                "x": 0,
                "shape_z": 1,
                "shape_y": 3,
                "shape_x": 4,
                "pixels": [100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123],
            },
        ],
        "planes": [
            {
                "z": 0,
                "t": 0,
                "c": 0,
                # 4*3 == 12 pixels (uint16 domain values)
                "pixels": [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
            },
            {
                "z": 0,
                "t": 0,
                "c": 1,
                "pixels": [100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123],
            },
        ],
        "masks": None,  # pa.null()
    }
