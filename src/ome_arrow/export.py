"""
Module for exporting OME-Arrow data to other formats.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ome_arrow.meta import OME_ARROW_STRUCT, OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION


def to_numpy(
    data: Dict[str, Any] | pa.StructScalar,
    dtype: np.dtype = np.uint16,
    strict: bool = True,
    clamp: bool = False,
) -> np.ndarray:
    """
    Convert an OME-Arrow record into a NumPy array shaped (T,C,Z,Y,X).

    The OME-Arrow "planes" are flattened YX slices indexed by (z, t, c).
    When chunks are present, this function reconstitutes the dense TCZYX array
    from chunked pixels instead of planes.

    Args:
        data:
            OME-Arrow data as a Python dict or a `pa.StructScalar`.
        dtype:
            Output dtype (default: np.uint16). If different from plane
            values, a cast (and optional clamp) is applied.
        strict:
            When True, raise if a plane has wrong pixel length. When
            False, truncate/pad that plane to the expected length.
        clamp:
            If True, clamp values to the valid range of the target
            dtype before casting.

    Returns:
        np.ndarray: Dense array with shape (T, C, Z, Y, X).

    Raises:
        KeyError: If required OME-Arrow fields are missing.
        ValueError: If dimensions are invalid or planes are malformed.

    Examples:
        >>> arr = ome_arrow_to_tczyx(my_row)  # (T, C, Z, Y, X)
        >>> arr.shape
        (1, 2, 1, 512, 512)
    """
    # Unwrap Arrow scalar to plain Python dict if needed.
    if isinstance(data, pa.StructScalar):
        data = data.as_py()

    pm = data["pixels_meta"]
    sx, sy = int(pm["size_x"]), int(pm["size_y"])
    sz, sc, st = int(pm["size_z"]), int(pm["size_c"]), int(pm["size_t"])
    if sx <= 0 or sy <= 0 or sz <= 0 or sc <= 0 or st <= 0:
        raise ValueError("All size_* fields must be positive integers.")

    expected_plane_len = sx * sy

    # Prepare target array (T,C,Z,Y,X), zero-filled by default.
    out = np.zeros((st, sc, sz, sy, sx), dtype=dtype)

    # Helper: cast (with optional clamp) to the output dtype.
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo, hi = info.min, info.max
    elif np.issubdtype(dtype, np.floating):
        lo, hi = -np.inf, np.inf
    else:
        # Rare dtypes: no clamping logic; rely on astype.
        lo, hi = -np.inf, np.inf

    def _cast_plane(a: np.ndarray) -> np.ndarray:
        if clamp:
            a = np.clip(a, lo, hi)
        return a.astype(dtype, copy=False)

    chunks = data.get("chunks") or []
    if chunks:
        chunk_grid = data.get("chunk_grid") or {}
        chunk_order = str(chunk_grid.get("chunk_order") or "ZYX").upper()
        if chunk_order != "ZYX":
            raise ValueError("Only chunk_order='ZYX' is supported for now.")

        for i, ch in enumerate(chunks):
            t = int(ch["t"])
            c = int(ch["c"])
            z = int(ch["z"])
            y = int(ch["y"])
            x = int(ch["x"])
            shape_z = int(ch["shape_z"])
            shape_y = int(ch["shape_y"])
            shape_x = int(ch["shape_x"])

            if not (0 <= t < st and 0 <= c < sc and 0 <= z < sz):
                raise ValueError(
                    f"chunks[{i}] index out of range: (t,c,z)=({t},{c},{z})"
                )
            if y < 0 or x < 0 or shape_z <= 0 or shape_y <= 0 or shape_x <= 0:
                raise ValueError(f"chunks[{i}] has invalid shape or origin.")

            pix = ch["pixels"]
            try:
                n = len(pix)
            except Exception as e:
                raise ValueError(f"chunks[{i}].pixels is not a sequence") from e

            expected_len = shape_z * shape_y * shape_x
            if n != expected_len:
                if strict:
                    raise ValueError(
                        f"chunks[{i}].pixels length {n} != expected {expected_len}"
                    )
                if n > expected_len:
                    pix = pix[:expected_len]
                else:
                    pix = list(pix) + [0] * (expected_len - n)

            arr3d = np.asarray(pix).reshape(shape_z, shape_y, shape_x)
            arr3d = _cast_plane(arr3d)
            out[t, c, z : z + shape_z, y : y + shape_y, x : x + shape_x] = arr3d
        return out

    # Fill planes.
    for i, p in enumerate(data.get("planes", [])):
        z = int(p["z"])
        t = int(p["t"])
        c = int(p["c"])

        if not (0 <= z < sz and 0 <= t < st and 0 <= c < sc):
            raise ValueError(f"planes[{i}] index out of range: (z,t,c)=({z},{t},{c})")

        pix = p["pixels"]
        # Ensure sequence-like and correct length.
        try:
            n = len(pix)
        except Exception as e:
            raise ValueError(f"planes[{i}].pixels is not a sequence") from e

        if n != expected_plane_len:
            if strict:
                raise ValueError(
                    f"planes[{i}].pixels length {n} != size_x*size_y {expected_plane_len}"
                )
            # Lenient mode: fix length by truncation or zero-pad.
            if n > expected_plane_len:
                pix = pix[:expected_plane_len]
            else:
                pix = list(pix) + [0] * (expected_plane_len - n)

        # Reshape to (Y,X) and cast.
        arr2d = np.asarray(pix).reshape(sy, sx)
        arr2d = _cast_plane(arr2d)
        out[t, c, z] = arr2d

    return out


def to_ome_tiff(
    data: Dict[str, Any] | pa.StructScalar,
    out_path: str,
    *,
    dtype: np.dtype = np.uint16,
    clamp: bool = False,
    dim_order: str = "TCZYX",
    compression: Optional[str] = "zlib",  # "zlib","lzma","jpegxl", or None
    compression_level: int = 6,
    tile: Optional[Tuple[int, int]] = None,  # (Y, X)
    use_channel_colors: bool = False,
) -> None:
    """
    Export an OME-Arrow record to OME-TIFF using BioIO's OmeTiffWriter.

    Notes
    -----
    - No 'bigtiff' kwarg is passed (invalid for tifffile.TiffWriter.write()).
      BigTIFF selection is automatic based on file size.
    """
    from ome_arrow.export import to_numpy  # your existing function

    try:
        from bioio.writers import OmeTiffWriter
    except Exception:
        from bioio_ome_tiff.writers import OmeTiffWriter  # type: ignore

    # PhysicalPixelSizes (robust import or shim)
    try:
        from bioio import PhysicalPixelSizes  # modern bioio
    except Exception:
        try:
            from bioio.types import PhysicalPixelSizes
        except Exception:
            try:
                from aicsimageio.types import PhysicalPixelSizes
            except Exception:
                from typing import NamedTuple
                from typing import Optional as _Opt

                class PhysicalPixelSizes(NamedTuple):  # type: ignore
                    Z: _Opt[float] = None
                    Y: _Opt[float] = None
                    X: _Opt[float] = None

    # 1) Dense array (T,C,Z,Y,X)
    arr = to_numpy(data, dtype=dtype, clamp=clamp)

    # 2) Metadata
    row = data.as_py() if isinstance(data, pa.StructScalar) else data
    pm = row["pixels_meta"]
    _st, sc, _sz, _sy, _sx = arr.shape

    # Channel names
    chs: Sequence[Dict[str, Any]] = pm.get("channels", []) or []
    channel_names = [f"C{i}" for i in range(sc)]
    if len(chs) == sc:
        for i, ch in enumerate(chs):
            nm = ch.get("name")
            if nm is not None:
                channel_names[i] = str(nm)

    # Optional channel colors (guarded)
    channel_colors_for_writer = None
    if use_channel_colors and len(chs) == sc:

        def _rgba_to_rgb(rgba: int) -> int:
            r = (rgba >> 24) & 0xFF
            g = (rgba >> 16) & 0xFF
            b = (rgba >> 8) & 0xFF
            return (r << 16) | (g << 8) | b

        flat_colors: list[int] = []
        for ch in chs:
            rgba = ch.get("color_rgba")
            flat_colors.append(
                _rgba_to_rgb(int(rgba)) if isinstance(rgba, int) else 0xFFFFFF
            )
        if len(flat_colors) == sc:
            channel_colors_for_writer = [flat_colors]  # list-per-image

    # Physical sizes (µm) in Z, Y, X order for BioIO
    p_dx = float(pm.get("physical_size_x", 1.0) or 1.0)
    p_dy = float(pm.get("physical_size_y", 1.0) or 1.0)
    p_dz = float(pm.get("physical_size_z", 1.0) or 1.0)
    pps_list = [PhysicalPixelSizes(Z=p_dz, Y=p_dy, X=p_dx)]

    # tifffile passthrough (NO 'bigtiff' here)
    tifffile_kwargs: Dict[str, Any] = {}
    if compression is not None:
        tifffile_kwargs["compression"] = compression
        if compression == "zlib":
            tifffile_kwargs["compressionargs"] = {"level": int(compression_level)}
    if tile is not None:
        tifffile_kwargs["tile"] = (int(tile[0]), int(tile[1]))

    # list-per-image payloads
    data_list = [arr]
    dim_order_list = [dim_order]
    image_name_list = [str(row.get("name") or row.get("id") or "image")]
    ch_names_list = [channel_names]

    # 3) Write
    OmeTiffWriter.save(
        data_list,
        out_path,
        dim_order=dim_order_list,
        image_name=image_name_list,
        channel_names=ch_names_list,
        channel_colors=channel_colors_for_writer,  # None or [flat list len=sc]
        physical_pixel_sizes=pps_list,
        tifffile_kwargs=tifffile_kwargs,
    )


def to_ome_zarr(
    data: Dict[str, Any] | pa.StructScalar,
    out_path: str,
    *,
    dtype: np.dtype = np.uint16,
    clamp: bool = False,
    # Axes order for the on-disk array — must match arr shape (T,C,Z,Y,X)
    dim_order: str = "TCZYX",
    # NGFF / multiscale
    multiscale_levels: int = 1,  # 1 = no pyramid; >1 builds levels
    downscale_spatial_by: int = 2,  # per-level factor for Z,Y,X
    zarr_format: int = 3,  # 3 (NGFF 0.5) or 2 (NGFF 0.4)
    # Storage knobs
    chunks: Optional[Tuple[int, int, int, int, int]] = None,  # (T,C,Z,Y,X) or None
    shards: Optional[Tuple[int, int, int, int, int]] = None,  # v3 only, optional
    compressor: Optional[str] = "zstd",  # "zstd","lz4","gzip", or None
    compressor_level: int = 3,
    # Optional display metadata (carried through if you later enrich channels/rdefs)
    image_name: Optional[str] = None,
) -> None:
    """
    Write OME-Zarr using your `OMEZarrWriter` (instance API).

    - Builds arr as (T,C,Z,Y,X) using your `to_numpy`.
    - Creates level shapes for a multiscale pyramid (if multiscale_levels>1).
    - Chooses Blosc codec compatible with zarr_format (v2 vs v3).
    - Populates axes names/types/units and physical pixel sizes from pixels_meta.
    - Uses default TCZYX chunks if none are provided.
    """
    # --- local import to avoid hard deps at module import time
    # Use the class you showed
    from bioio_ome_zarr.writers import OMEZarrWriter

    from ome_arrow.export import to_numpy  # your existing function

    # Optional compressors for v2 vs v3
    compressor_obj = None
    if compressor is not None:
        if zarr_format == 2:
            # numcodecs Blosc (v2 path)
            from numcodecs import Blosc as BloscV2

            cname = {"zstd": "zstd", "lz4": "lz4", "gzip": "zlib"}.get(
                compressor, "zstd"
            )
            compressor_obj = BloscV2(
                cname=cname, clevel=int(compressor_level), shuffle=BloscV2.BITSHUFFLE
            )
        else:
            # zarr v3 codec
            from zarr.codecs import BloscCodec, BloscShuffle

            cname = {"zstd": "zstd", "lz4": "lz4", "gzip": "zlib"}.get(
                compressor, "zstd"
            )
            compressor_obj = BloscCodec(
                cname=cname,
                clevel=int(compressor_level),
                shuffle=BloscShuffle.bitshuffle,
            )

    # 1) Dense pixel data (T,C,Z,Y,X)
    arr = to_numpy(data, dtype=dtype, clamp=clamp)

    # 2) Unwrap OME-Arrow metadata
    row = data.as_py() if isinstance(data, pa.StructScalar) else data
    pm = row["pixels_meta"]
    st, sc, sz, sy, sx = arr.shape

    # 3) Axis metadata (names/types/units aligned with T,C,Z,Y,X)
    axes_names = [a.lower() for a in dim_order]  # ["t","c","z","y","x"]
    axes_types = ["time", "channel", "space", "space", "space"]
    # Units: micrometers for spatial, leave T/C None
    axes_units = [
        None,
        None,
        pm.get("physical_size_z_unit") or "µm",
        pm.get("physical_size_y_unit") or "µm",
        pm.get("physical_size_x_unit") or "µm",
    ]
    # Physical pixel sizes at level 0 in axis order
    p_dx = float(pm.get("physical_size_x", 1.0) or 1.0)
    p_dy = float(pm.get("physical_size_y", 1.0) or 1.0)
    p_dz = float(pm.get("physical_size_z", 1.0) or 1.0)
    physical_pixel_size = [1.0, 1.0, p_dz, p_dy, p_dx]  # T,C,Z,Y,X

    # 4) Multiscale level shapes (level 0 first). Only spatial dims are downscaled.
    def _down(a: int, f: int) -> int:
        return max(1, a // f)

    def _default_chunks_tcxyz(
        shape: Tuple[int, int, int, int, int],
    ) -> Tuple[int, int, int, int, int]:
        t, c, z, y, x = shape
        cz = min(z, 4) if z > 1 else 1
        cy = min(y, 512)
        cx = min(x, 512)
        return (1, 1, cz, cy, cx)

    def _level_shapes_tcxyz(levels: int) -> List[Tuple[int, int, int, int, int]]:
        shapes = [(st, sc, sz, sy, sx)]
        for _ in range(levels - 1):
            t, c, z, y, x = shapes[-1]
            shapes.append(
                (
                    t,
                    c,
                    _down(z, downscale_spatial_by),
                    _down(y, downscale_spatial_by),
                    _down(x, downscale_spatial_by),
                )
            )
        return shapes

    multiscale_levels = max(1, int(multiscale_levels))
    level_shapes: List[Tuple[int, int, int, int, int]] = _level_shapes_tcxyz(
        multiscale_levels
    )

    # 5) Chunking / shards (can be single-shape or per-level;
    # we pass single-shape if provided)
    chunk_shape: Optional[List[Tuple[int, ...]]] = None
    if chunks is None:
        chunks = _default_chunks_tcxyz((st, sc, sz, sy, sx))
    if chunks is not None:
        chunk_shape = [tuple(int(v) for v in chunks)] * multiscale_levels

    shard_shape: Optional[List[Tuple[int, ...]]] = None
    if shards is not None and zarr_format == 3:
        shard_shape = [tuple(int(v) for v in shards)] * multiscale_levels

    # 6) Image name default
    img_name = image_name or str(row.get("name") or row.get("id") or "Image")

    # 7) Instantiate writer with your class constructor
    writer = OMEZarrWriter(
        store=out_path,
        level_shapes=level_shapes,
        dtype=dtype,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compressor=compressor_obj,
        zarr_format=3 if int(zarr_format) == 3 else 2,
        image_name=img_name,
        channels=None,  # you can map your channel metadata here later
        rdefs=None,  # optional OMERO display metadata
        creator_info=None,  # optional "creator" block
        root_transform=None,  # optional NGFF root transform
        axes_names=axes_names,
        axes_types=axes_types,
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
    )

    # 8) Write full-resolution; writer will build & fill lower levels
    writer.write_full_volume(arr)


def to_ome_parquet(
    data: Dict[str, Any] | pa.StructScalar,
    out_path: str,
    column_name: str = "image",
    file_metadata: Optional[Dict[str, str]] = None,
    compression: Optional[str] = "zstd",
    row_group_size: Optional[int] = None,
) -> None:
    """
    Export an OME-Arrow record to a Parquet file as a single-row, single-column table.
    The single column holds a struct with the OME-Arrow schema.
    """

    # 1) Normalize to a plain Python dict (works better with pyarrow builders,
    #    especially when the struct has a `null`-typed field like "masks").
    if isinstance(data, pa.StructScalar):
        record_dict = data.as_py()
    else:
        # Validate by round-tripping through a typed scalar, then back to dict.
        record_dict = {f.name: data.get(f.name) for f in OME_ARROW_STRUCT}
        record_dict = pa.scalar(record_dict, type=OME_ARROW_STRUCT).as_py()

    # 2) Build a single-row struct array from the dict, explicitly passing the schema
    struct_array = pa.array([record_dict], type=OME_ARROW_STRUCT)  # len=1

    # 3) Wrap into a one-column table
    table = pa.table({column_name: struct_array})

    # 4) Attach optional file-level metadata
    meta: Dict[bytes, bytes] = dict(table.schema.metadata or {})
    try:
        meta[b"ome.arrow.type"] = str(OME_ARROW_TAG_TYPE).encode("utf-8")
        meta[b"ome.arrow.version"] = str(OME_ARROW_TAG_VERSION).encode("utf-8")
    except Exception:
        pass
    if file_metadata:
        for k, v in file_metadata.items():
            meta[str(k).encode("utf-8")] = str(v).encode("utf-8")
    table = table.replace_schema_metadata(meta)

    # 5) Write Parquet (single row, single column)
    pq.write_table(
        table,
        out_path,
        compression=compression,
        row_group_size=row_group_size,
    )


def to_ome_vortex(
    data: Dict[str, Any] | pa.StructScalar,
    out_path: str,
    column_name: str = "image",
    file_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Export an OME-Arrow record to a Vortex file.

    The file is written as a single-row, single-column Arrow table where the
    column holds a struct with the OME-Arrow schema.

    Args:
        data: OME-Arrow dict or StructScalar.
        out_path: Output path for the Vortex file.
        column_name: Column name to store the struct.
        file_metadata: Optional file-level metadata to attach.

    Raises:
        ImportError: If the optional `vortex-data` dependency is missing.
    """

    try:
        import vortex.io as vxio
    except ImportError as exc:
        raise ImportError(
            "Vortex export requires the optional 'vortex-data' dependency."
        ) from exc

    # 1) Normalize to a plain Python dict (works better with pyarrow builders,
    #    especially when the struct has a `null`-typed field like "masks").
    if isinstance(data, pa.StructScalar):
        record_dict = data.as_py()
    else:
        # Validate by round-tripping through a typed scalar, then back to dict.
        record_dict = {f.name: data.get(f.name) for f in OME_ARROW_STRUCT}
        record_dict = pa.scalar(record_dict, type=OME_ARROW_STRUCT).as_py()

    # 2) Build a single-row struct array from the dict, explicitly passing the schema
    struct_array = pa.array([record_dict], type=OME_ARROW_STRUCT)  # len=1

    # 3) Wrap into a one-column table
    table = pa.table({column_name: struct_array})

    # 4) Attach optional file-level metadata
    meta: Dict[bytes, bytes] = dict(table.schema.metadata or {})
    try:
        meta[b"ome.arrow.type"] = str(OME_ARROW_TAG_TYPE).encode("utf-8")
        meta[b"ome.arrow.version"] = str(OME_ARROW_TAG_VERSION).encode("utf-8")
    except Exception:
        pass
    if file_metadata:
        for k, v in file_metadata.items():
            meta[str(k).encode("utf-8")] = str(v).encode("utf-8")
    table = table.replace_schema_metadata(meta)

    # 5) Write Vortex (single row, single column)
    vxio.write(table, str(out_path))
