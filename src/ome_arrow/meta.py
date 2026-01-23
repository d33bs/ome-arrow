"""
Meta-definition for OME-Arrow format.
"""

import pyarrow as pa

from ._version import version as ome_arrow_version

OME_ARROW_TAG_TYPE = "ome.arrow"
OME_ARROW_TAG_VERSION = ome_arrow_version

# OME_ARROW_STRUCT: ome-arrow record (describes one image/value).
#  - type/version: quick identity & evolution.
#  - id/name/acquisition_datetime: identity & provenance.
#  - image_type: open-ended image kind (e.g., "image", "label").
#  - pixels_meta: pixels struct (sizes, units, channels).
#  - planes: list of planes struct entries, one per (t,c,z).
#  - chunk_grid/chunks: optional chunked pixels (TCZYX-aware), stored as Arrow lists.
#  - masks: reserved for future labels/ROIs (placeholder).
OME_ARROW_STRUCT: pa.StructType = pa.struct(
    [
        pa.field("type", pa.string()),  # must be "ome.arrow"
        pa.field("version", pa.string()),  # e.g., "1.0.0"
        pa.field("id", pa.string()),  # stable image identifier
        pa.field("name", pa.string()),  # human label
        pa.field("image_type", pa.string()),  # open-ended (e.g., "image", "label")
        pa.field("acquisition_datetime", pa.timestamp("us")),
        # PIXELS: OME-like "Pixels" header summarizing shape & scale.
        #  - dimension_order: hint like "XYZCT" (or "XYCT" when Z==1).
        #  - type: numeric storage type (e.g., "uint16").
        #  - size_*: axis lengths.
        #  - physical_size_* (+ *_unit): microscope scale in micrometers.
        #  - channels: list of channel struct entries (one per channel).
        pa.field(
            "pixels_meta",
            pa.struct(
                [
                    pa.field("dimension_order", pa.string()),  # "XYZCT" / "XYCT"
                    pa.field("type", pa.string()),  # "uint8","uint16","float",...
                    pa.field("size_x", pa.int32()),  # width  (pixels)
                    pa.field("size_y", pa.int32()),  # height (pixels)
                    pa.field("size_z", pa.int32()),  # z-slices (1 for 2D)
                    pa.field("size_c", pa.int16()),  # channels
                    pa.field("size_t", pa.int32()),  # time points
                    pa.field("physical_size_x", pa.float32()),  # µm per pixel (X)
                    pa.field("physical_size_y", pa.float32()),  # µm per pixel (Y)
                    pa.field("physical_size_z", pa.float32()),  # µm per z-step
                    pa.field("physical_size_x_unit", pa.string()),  # usually "µm"
                    pa.field("physical_size_y_unit", pa.string()),
                    pa.field("physical_size_z_unit", pa.string()),
                    pa.field(
                        "channels",
                        pa.list_(
                            # CHANNELS: one entry per channel (e.g., DNA, Mito, ER).
                            #  - emission_um / excitation_um: wavelengths (micrometers).
                            #  - illumination: modality (e.g., "Epifluorescence").
                            #  - color_rgba: preferred display color
                            #       (packed 0xRRGGBBAA).
                            pa.struct(
                                [
                                    pa.field("id", pa.string()),
                                    pa.field("name", pa.string()),
                                    pa.field("emission_um", pa.float32()),
                                    pa.field("excitation_um", pa.float32()),
                                    pa.field("illumination", pa.string()),
                                    pa.field("color_rgba", pa.uint32()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        ),
        # CHUNK GRID: optional chunking metadata for random access.
        #  - order: axis order for the full array, e.g., "TCZYX".
        #  - chunk_*: chunk sizes for each axis (defaults to 1 for T/C).
        #  - chunk_order: order used to flatten chunk pixels (default "ZYX").
        pa.field(
            "chunk_grid",
            pa.struct(
                [
                    pa.field("order", pa.string()),
                    pa.field("chunk_t", pa.int32()),
                    pa.field("chunk_c", pa.int16()),
                    pa.field("chunk_z", pa.int32()),
                    pa.field("chunk_y", pa.int32()),
                    pa.field("chunk_x", pa.int32()),
                    pa.field("chunk_order", pa.string()),
                ]
            ),
        ),
        # CHUNKS: list of chunk entries (Arrow-native, no binary payloads).
        #  - pixels flattened in chunk_order (default "ZYX").
        pa.field(
            "chunks",
            pa.list_(
                pa.struct(
                    [
                        pa.field("t", pa.int32()),
                        pa.field("c", pa.int16()),
                        pa.field("z", pa.int32()),
                        pa.field("y", pa.int32()),
                        pa.field("x", pa.int32()),
                        pa.field("shape_z", pa.int32()),
                        pa.field("shape_y", pa.int32()),
                        pa.field("shape_x", pa.int32()),
                        pa.field("pixels", pa.list_(pa.uint16())),
                    ]
                )
            ),
        ),
        # PLANES: one 2D image plane for a specific (t, c, z).
        #  - pixels: flattened numeric list (Y*X) for analysis-ready computation.
        pa.field(
            "planes",
            pa.list_(
                pa.struct(
                    [
                        pa.field("z", pa.int32()),
                        pa.field("t", pa.int32()),
                        pa.field("c", pa.int16()),
                        pa.field(
                            "pixels", pa.list_(pa.uint16())
                        ),  # keep numeric (not PNG/JPEG)
                    ]
                )
            ),
        ),
        pa.field("masks", pa.null()),  # reserved for future annotations
    ]
)
