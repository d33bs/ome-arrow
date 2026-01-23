"""
Tests for chunked pixel support.
"""

import numpy as np

from ome_arrow.export import to_numpy
from ome_arrow.ingest import to_ome_arrow


def test_to_numpy_from_chunks(example_correct_data: dict) -> None:
    """Reconstruct dense arrays from chunked pixels."""
    data = dict(example_correct_data)
    data["planes"] = []

    arr = to_numpy(data)

    assert arr.shape == (1, 2, 1, 3, 4)
    np.testing.assert_array_equal(
        arr[0, 0, 0],
        np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]),
    )
    np.testing.assert_array_equal(
        arr[0, 1, 0],
        np.array([[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]]),
    )


def test_to_ome_arrow_builds_chunks() -> None:
    """Build chunked pixels from planes when requested."""
    planes = [
        {
            "z": 0,
            "t": 0,
            "c": 0,
            "pixels": [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        }
    ]

    scalar = to_ome_arrow(
        image_id="img-0002",
        name="Chunky",
        image_type="image",
        dimension_order="XYCT",
        dtype="uint16",
        size_x=4,
        size_y=3,
        size_z=1,
        size_c=1,
        size_t=1,
        channels=[{"id": "ch-0", "name": "C0"}],
        planes=planes,
        chunk_shape=(1, 2, 2),
        build_chunks=True,
    )

    record = scalar.as_py()
    assert record["chunk_grid"]["chunk_y"] == 2
    assert record["chunk_grid"]["chunk_x"] == 2
    assert len(record["chunks"]) == 4

    first_chunk = record["chunks"][0]
    assert first_chunk["t"] == 0
    assert first_chunk["c"] == 0
    assert first_chunk["z"] == 0
    assert first_chunk["y"] == 0
    assert first_chunk["x"] == 0
    assert first_chunk["shape_y"] == 2
    assert first_chunk["shape_x"] == 2
    assert first_chunk["pixels"] == [0, 1, 10, 11]
