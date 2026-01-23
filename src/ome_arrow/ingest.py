"""
Converting to and from OME-Arrow formats.
"""

import itertools
import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bioio_ome_tiff
import bioio_tifffile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from bioio import BioImage
from bioio_ome_zarr import Reader as OMEZarrReader

from ome_arrow.meta import OME_ARROW_STRUCT, OME_ARROW_TAG_TYPE, OME_ARROW_TAG_VERSION


def _ome_arrow_from_table(
    table: pa.Table,
    *,
    column_name: Optional[str],
    row_index: int,
    strict_schema: bool,
) -> pa.StructScalar:
    """Extract a single OME-Arrow record from an Arrow table.

    Args:
        table: Source Arrow table.
        column_name: Column to read; auto-detected when None or invalid.
        row_index: Row index to extract.
        strict_schema: Require the exact OME-Arrow schema if True.

    Returns:
        A typed OME-Arrow StructScalar.

    Raises:
        ValueError: If the row index is out of range or no suitable column exists.
    """
    if table.num_rows == 0:
        raise ValueError("Table contains 0 rows; expected at least 1.")
    if not (0 <= row_index < table.num_rows):
        raise ValueError(f"row_index {row_index} out of range [0, {table.num_rows}).")

    # 1) Locate the OME-Arrow column
    def _struct_matches_ome_fields(t: pa.StructType) -> bool:
        ome_fields = {f.name for f in OME_ARROW_STRUCT}
        required_fields = ome_fields - {"image_type", "chunk_grid", "chunks"}
        col_fields = {f.name for f in t}
        return required_fields.issubset(col_fields)

    requested_name = column_name
    candidate_col = None
    autodetected_name = None

    if column_name is not None and column_name in table.column_names:
        arr = table[column_name]
        if not pa.types.is_struct(arr.type):
            raise ValueError(f"Column '{column_name}' is not a Struct; got {arr.type}.")
        if strict_schema and arr.type != OME_ARROW_STRUCT:
            raise ValueError(
                f"Column '{column_name}' schema != OME_ARROW_STRUCT.\n"
                f"Got:   {arr.type}\n"
                f"Expect:{OME_ARROW_STRUCT}"
            )
        if not strict_schema and not _struct_matches_ome_fields(arr.type):
            raise ValueError(
                f"Column '{column_name}' does not have the expected OME-Arrow fields."
            )
        candidate_col = arr
    else:
        # Auto-detect a struct column that matches OME-Arrow fields
        for name in table.column_names:
            arr = table[name]
            if pa.types.is_struct(arr.type):
                if strict_schema and arr.type == OME_ARROW_STRUCT:
                    candidate_col = arr
                    autodetected_name = name
                    column_name = name
                    break
                if not strict_schema and _struct_matches_ome_fields(arr.type):
                    candidate_col = arr
                    autodetected_name = name
                    column_name = name
                    break
        if candidate_col is None:
            if column_name is None:
                hint = "no struct column with OME-Arrow fields was found."
            else:
                hint = f"column '{column_name}' not found and auto-detection failed."
            raise ValueError(f"Could not locate an OME-Arrow struct column: {hint}")

    # Emit warning if auto-detection was used
    if autodetected_name is not None and autodetected_name != requested_name:
        warnings.warn(
            f"Requested column '{requested_name}' was not usable or not found. "
            f"Auto-detected OME-Arrow column '{autodetected_name}'.",
            UserWarning,
            stacklevel=2,
        )

    # 2) Extract the row as a Python dict
    record_dict: Dict[str, Any] = candidate_col.slice(row_index, 1).to_pylist()[0]
    # Back-compat: older files won't include image_type; default to None.
    if "image_type" not in record_dict:
        record_dict["image_type"] = None
    # Drop unexpected fields before casting to the canonical schema.
    record_dict = {f.name: record_dict.get(f.name) for f in OME_ARROW_STRUCT}

    # 3) Reconstruct a typed StructScalar using the canonical schema
    scalar = pa.scalar(record_dict, type=OME_ARROW_STRUCT)

    # Optional: soft validation via file-level metadata (if present)
    try:
        meta = table.schema.metadata or {}
        meta.get(b"ome.arrow.type", b"").decode() == str(OME_ARROW_TAG_TYPE)
        meta.get(b"ome.arrow.version", b"").decode() == str(OME_ARROW_TAG_VERSION)
    except Exception:
        pass

    return scalar


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip().lower()
    if u in {"micrometer", "micrometre", "micron", "microns", "um", "µm"}:
        return "µm"
    if u in {"nanometer", "nanometre", "nm"}:
        return "nm"
    return unit


def _read_physical_pixel_sizes(
    img: BioImage,
) -> tuple[float, float, float, str | None, bool]:
    pps = getattr(img, "physical_pixel_sizes", None)
    if pps is None:
        return 1.0, 1.0, 1.0, None, False

    vx = getattr(pps, "X", None) or getattr(pps, "x", None)
    vy = getattr(pps, "Y", None) or getattr(pps, "y", None)
    vz = getattr(pps, "Z", None) or getattr(pps, "z", None)

    if vx is None and vy is None and vz is None:
        return 1.0, 1.0, 1.0, None, False

    try:
        psize_x = float(vx or 1.0)
        psize_y = float(vy or 1.0)
        psize_z = float(vz or 1.0)
    except Exception:
        return 1.0, 1.0, 1.0, None, False

    unit = getattr(pps, "unit", None) or getattr(pps, "units", None)
    unit = _normalize_unit(str(unit)) if unit is not None else None

    return psize_x, psize_y, psize_z, unit, True


def _load_zarr_attrs(zarr_path: Path) -> dict:
    zarr_json = zarr_path / "zarr.json"
    if zarr_json.exists():
        try:
            data = json.loads(zarr_json.read_text())
            return data.get("attributes") or data.get("attrs") or {}
        except Exception:
            return {}
    zattrs = zarr_path / ".zattrs"
    if zattrs.exists():
        try:
            return json.loads(zattrs.read_text())
        except Exception:
            return {}
    return {}


def _extract_multiscales(attrs: dict) -> list[dict]:
    if not isinstance(attrs, dict):
        return []
    ome = attrs.get("ome")
    if isinstance(ome, dict) and isinstance(ome.get("multiscales"), list):
        return ome["multiscales"]
    if isinstance(attrs.get("multiscales"), list):
        return attrs["multiscales"]
    return []


def _read_ngff_scale(zarr_path: Path) -> tuple[float, float, float, str | None] | None:
    zarr_root = zarr_path
    for parent in [zarr_path, *list(zarr_path.parents)]:
        if parent.suffix.lower() in {".zarr", ".ome.zarr"}:
            zarr_root = parent
            break

    for candidate in (zarr_path, zarr_root):
        attrs = _load_zarr_attrs(candidate)
        multiscales = _extract_multiscales(attrs)
        if multiscales:
            break
    else:
        return None

    ms = multiscales[0]
    axes = ms.get("axes") or []
    datasets = ms.get("datasets") or []
    if not axes or not datasets:
        return None

    ds = next((d for d in datasets if str(d.get("path")) == "0"), datasets[0])
    cts = ds.get("coordinateTransformations") or []
    scale_ct = next((ct for ct in cts if ct.get("type") == "scale"), None)
    if not scale_ct:
        return None

    scale = scale_ct.get("scale") or []
    if len(scale) != len(axes):
        return None

    axis_scale: dict[str, float] = {}
    axis_unit: dict[str, str] = {}
    for i, ax in enumerate(axes):
        name = str(ax.get("name", "")).lower()
        if name in {"x", "y", "z"}:
            try:
                axis_scale[name] = float(scale[i])
            except Exception:
                continue
            unit = _normalize_unit(ax.get("unit"))
            if unit:
                axis_unit[name] = unit

    if not axis_scale:
        return None

    psize_x = axis_scale.get("x", 1.0)
    psize_y = axis_scale.get("y", 1.0)
    psize_z = axis_scale.get("z", 1.0)

    units = [axis_unit.get(a) for a in ("x", "y", "z") if axis_unit.get(a)]
    unit = units[0] if units and len(set(units)) == 1 else None

    return psize_x, psize_y, psize_z, unit


def _normalize_chunk_shape(
    chunk_shape: Optional[Tuple[int, int, int]],
    size_z: int,
    size_y: int,
    size_x: int,
) -> Tuple[int, int, int]:
    """Normalize a chunk shape against image bounds.

    Args:
        chunk_shape: Desired chunk shape as (Z, Y, X), or None.
        size_z: Total Z size of the image.
        size_y: Total Y size of the image.
        size_x: Total X size of the image.

    Returns:
        Tuple[int, int, int]: Normalized (Z, Y, X) chunk shape.
    """
    if chunk_shape is None:
        chunk_shape = (1, 512, 512)
    cz = max(1, min(int(chunk_shape[0]), int(size_z)))
    cy = max(1, min(int(chunk_shape[1]), int(size_y)))
    cx = max(1, min(int(chunk_shape[2]), int(size_x)))
    return cz, cy, cx


def _build_chunks_from_planes(
    *,
    planes: List[Dict[str, Any]],
    size_t: int,
    size_c: int,
    size_z: int,
    size_y: int,
    size_x: int,
    chunk_shape: Optional[Tuple[int, int, int]],
    chunk_order: str = "ZYX",
) -> List[Dict[str, Any]]:
    """Build chunked pixels from a list of flattened planes.

    Args:
        planes: List of plane dicts with keys z, t, c, and pixels.
        size_t: Total T size of the image.
        size_c: Total C size of the image.
        size_z: Total Z size of the image.
        size_y: Total Y size of the image.
        size_x: Total X size of the image.
        chunk_shape: Desired chunk shape as (Z, Y, X).
        chunk_order: Flattening order for chunk pixels (default "ZYX").

    Returns:
        List[Dict[str, Any]]: Chunk list with pixels stored as flat lists.

    Raises:
        ValueError: If an unsupported chunk_order is requested.
    """
    if str(chunk_order).upper() != "ZYX":
        raise ValueError("Only chunk_order='ZYX' is supported for now.")

    cz, cy, cx = _normalize_chunk_shape(chunk_shape, size_z, size_y, size_x)

    plane_map: Dict[Tuple[int, int, int], np.ndarray] = {}
    for p in planes:
        z = int(p["z"])
        t = int(p["t"])
        c = int(p["c"])
        pix = p["pixels"]
        arr2d = np.asarray(pix).reshape(size_y, size_x)
        plane_map[(t, c, z)] = arr2d

    chunks: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z0 in range(0, size_z, cz):
                sz = min(cz, size_z - z0)
                for y0 in range(0, size_y, cy):
                    sy = min(cy, size_y - y0)
                    for x0 in range(0, size_x, cx):
                        sx = min(cx, size_x - x0)
                        slab = np.zeros((sz, sy, sx), dtype=np.uint16)
                        for zi in range(sz):
                            plane = plane_map.get((t, c, z0 + zi))
                            if plane is None:
                                continue
                            slab[zi] = plane[y0 : y0 + sy, x0 : x0 + sx]
                        chunks.append(
                            {
                                "t": t,
                                "c": c,
                                "z": z0,
                                "y": y0,
                                "x": x0,
                                "shape_z": sz,
                                "shape_y": sy,
                                "shape_x": sx,
                                "pixels": slab.reshape(-1).tolist(),
                            }
                        )
    return chunks


def to_ome_arrow(
    type_: str = OME_ARROW_TAG_TYPE,
    version: str = OME_ARROW_TAG_VERSION,
    image_id: str = "unnamed",
    name: str = "unknown",
    image_type: str | None = "image",
    acquisition_datetime: Optional[datetime] = None,
    dimension_order: str = "XYZCT",
    dtype: str = "uint16",
    size_x: int = 1,
    size_y: int = 1,
    size_z: int = 1,
    size_c: int = 1,
    size_t: int = 1,
    physical_size_x: float = 1.0,
    physical_size_y: float = 1.0,
    physical_size_z: float = 1.0,
    physical_size_unit: str = "µm",
    channels: Optional[List[Dict[str, Any]]] = None,
    planes: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = (1, 512, 512),  # (Z, Y, X)
    chunk_order: str = "ZYX",
    build_chunks: bool = True,
    masks: Any = None,
) -> pa.StructScalar:
    """
    Create a typed OME-Arrow StructScalar with sensible defaults.

    This builds and validates a nested dict that conforms to the given
    StructType (e.g., OME_ARROW_STRUCT). You can override any field
    explicitly; others use safe defaults.

    Args:
        type_: Top-level type string ("ome.arrow" by default).
        version: Specification version string.
        image_id: Unique image identifier.
        name: Human-friendly name.
        image_type: Open-ended image kind (e.g., "image", "label"). Note that
            from_* helpers pass image_type=None by default to preserve
            "unspecified" vs explicitly set ("image").
        acquisition_datetime: Datetime of acquisition (defaults to now).
        dimension_order: Dimension order ("XYZCT" or "XYCT").
        dtype: Pixel data type string (e.g., "uint16").
        size_x, size_y, size_z, size_c, size_t: Axis sizes.
        physical_size_x/y/z: Physical scaling in µm.
        physical_size_unit: Unit string, default "µm".
        channels: List of channel dicts. Autogenerates one if None.
        planes: List of plane dicts. Empty if None.
        chunks: Optional list of chunk dicts. If None and build_chunks is True,
            chunks are derived from planes using chunk_shape.
        chunk_shape: Chunk shape as (Z, Y, X). Defaults to (1, 512, 512).
        chunk_order: Flattening order for chunk pixels (default "ZYX").
        build_chunks: If True, build chunked pixels from planes when chunks
            is None.
        masks: Optional placeholder for future annotations.

    Returns:
        pa.StructScalar: A validated StructScalar for the schema.

    Example:
        >>> s = to_struct_scalar(OME_ARROW_STRUCT, image_id="img001")
        >>> s.type == OME_ARROW_STRUCT
        True
    """

    type_ = str(type_)
    version = str(version)
    image_id = str(image_id)
    name = str(name)
    image_type = None if image_type is None else str(image_type)
    dimension_order = str(dimension_order)
    dtype = str(dtype)
    physical_size_unit = str(physical_size_unit)

    # Sensible defaults for channels and planes
    if channels is None:
        channels = [
            {
                "id": "ch-0",
                "name": "default",
                "emission_um": 0.0,
                "excitation_um": 0.0,
                "illumination": "Unknown",
                "color_rgba": 0xFFFFFFFF,
            }
        ]
    else:
        # --- NEW: coerce channel text fields to str ------------------
        for ch in channels:
            if "id" in ch:
                ch["id"] = str(ch["id"])
            if "name" in ch:
                ch["name"] = str(ch["name"])
            if "illumination" in ch:
                ch["illumination"] = str(ch["illumination"])

    if planes is None:
        planes = [{"z": 0, "t": 0, "c": 0, "pixels": [0] * (size_x * size_y)}]

    if chunks is None and build_chunks:
        chunks = _build_chunks_from_planes(
            planes=planes,
            size_t=size_t,
            size_c=size_c,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
            chunk_shape=chunk_shape,
            chunk_order=chunk_order,
        )

    chunk_grid = None
    if chunks is not None:
        cz, cy, cx = _normalize_chunk_shape(chunk_shape, size_z, size_y, size_x)
        chunk_grid = {
            "order": "TCZYX",
            "chunk_t": 1,
            "chunk_c": 1,
            "chunk_z": cz,
            "chunk_y": cy,
            "chunk_x": cx,
            "chunk_order": str(chunk_order),
        }

    record = {
        "type": type_,
        "version": version,
        "id": image_id,
        "name": name,
        "image_type": image_type,
        "acquisition_datetime": acquisition_datetime or datetime.now(timezone.utc),
        "pixels_meta": {
            "dimension_order": dimension_order,
            "type": dtype,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "size_c": size_c,
            "size_t": size_t,
            "physical_size_x": physical_size_x,
            "physical_size_y": physical_size_y,
            "physical_size_z": physical_size_z,
            "physical_size_x_unit": physical_size_unit,
            "physical_size_y_unit": physical_size_unit,
            "physical_size_z_unit": physical_size_unit,
            "channels": channels,
        },
        "chunk_grid": chunk_grid,
        "chunks": chunks,
        "planes": planes,
        "masks": masks,
    }

    return pa.scalar(record, type=OME_ARROW_STRUCT)


def from_numpy(
    arr: np.ndarray,
    *,
    dim_order: str = "TCZYX",
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    image_type: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
    chunk_shape: Optional[Tuple[int, int, int]] = (1, 512, 512),
    chunk_order: str = "ZYX",
    build_chunks: bool = True,
    # meta
    physical_size_x: float = 1.0,
    physical_size_y: float = 1.0,
    physical_size_z: float = 1.0,
    physical_size_unit: str = "µm",
    dtype_meta: Optional[str] = None,  # if None, inferred from output dtype
) -> pa.StructScalar:
    """Build an OME-Arrow StructScalar from a NumPy array.

    Args:
        arr: Image data with axes described by `dim_order`.
        dim_order: Axis labels for `arr`. Must include "Y" and "X".
            Supported examples: "YX", "ZYX", "CYX", "CZYX", "TYX", "TCYX", "TCZYX".
        image_id: Optional stable image identifier.
        name: Optional human label.
        image_type: Open-ended image kind (e.g., "image", "label").
        channel_names: Names for channels; defaults to C0..C{n-1}.
        acquisition_datetime: Defaults to now (UTC) if None.
        clamp_to_uint16: If True, clamp/cast planes to uint16 before serialization.
        chunk_shape: Chunk shape as (Z, Y, X). Defaults to (1, 512, 512).
        chunk_order: Flattening order for chunk pixels (default "ZYX").
        build_chunks: If True, build chunked pixels from planes.
        physical_size_x: Spatial pixel size (µm) for X.
        physical_size_y: Spatial pixel size (µm) for Y.
        physical_size_z: Spatial pixel size (µm) for Z when present.
        physical_size_unit: Unit string for spatial axes (default "µm").
        dtype_meta: Pixel dtype string to place in metadata; if None, inferred
            from the (possibly cast) array's dtype.

    Returns:
        pa.StructScalar: Typed OME-Arrow record (schema = OME_ARROW_STRUCT).

    Raises:
        TypeError: If `arr` is not a NumPy ndarray.
        ValueError: If `dim_order` is invalid or dimensions are non-positive.

    Notes:
        - If Z is not in `dim_order`, `size_z` will be 1 and the meta
          dimension_order becomes "XYCT"; otherwise "XYZCT".
        - If T/C are absent in `dim_order`, they default to size 1.
    """

    if not isinstance(arr, np.ndarray):
        raise TypeError("from_numpy expects a NumPy ndarray.")

    dims = dim_order.upper()
    if "Y" not in dims or "X" not in dims:
        raise ValueError("dim_order must include 'Y' and 'X' axes.")

    # Map current axes -> indices
    axis_to_idx: Dict[str, int] = {ax: i for i, ax in enumerate(dims)}

    # Extract sizes with defaults for missing axes
    size_x = int(arr.shape[axis_to_idx["X"]])
    size_y = int(arr.shape[axis_to_idx["Y"]])
    size_z = int(arr.shape[axis_to_idx["Z"]]) if "Z" in axis_to_idx else 1
    size_c = int(arr.shape[axis_to_idx["C"]]) if "C" in axis_to_idx else 1
    size_t = int(arr.shape[axis_to_idx["T"]]) if "T" in axis_to_idx else 1

    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dimensions.")

    # Reorder to a standard (T, C, Z, Y, X) view for plane extraction
    desired_axes = ["T", "C", "Z", "Y", "X"]
    current_axes = list(dims)
    # Insert absent axes with size 1 using np.expand_dims
    view = arr
    for ax in desired_axes:
        if ax not in axis_to_idx:
            # Append a new singleton axis at the end, then we'll permute
            view = np.expand_dims(view, axis=-1)
            # Pretend this new axis now exists at the last index
            current_axes.append(ax)
            axis_to_idx = {a: i for i, a in enumerate(current_axes)}

    # Permute to TCZYX
    perm = [axis_to_idx[a] for a in desired_axes]
    tczyx = np.transpose(view, axes=perm)

    # Validate final shape
    if tuple(tczyx.shape) != (size_t, size_c, size_z, size_y, size_x):
        # This should not happen, but guard just in case
        raise ValueError(
            "Internal axis reordering mismatch: "
            f"got {tczyx.shape} vs expected {(size_t, size_c, size_z, size_y, size_x)}"
        )

    # Clamp/cast
    if clamp_to_uint16 and tczyx.dtype != np.uint16:
        tczyx = np.clip(tczyx, 0, 65535).astype(np.uint16, copy=False)

    # Channel names
    if not channel_names or len(channel_names) != size_c:
        channel_names = [f"C{i}" for i in range(size_c)]
    channel_names = [str(x) for x in channel_names]

    channels = [
        {
            "id": f"ch-{i}",
            "name": channel_names[i],
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    # Build planes: flatten YX per (t,c,z)
    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = tczyx[t, c, z]
                planes.append(
                    {"z": z, "t": t, "c": c, "pixels": plane.ravel().tolist()}
                )

    # Meta dimension_order: mirror your other ingests
    meta_dim_order = "XYCT" if size_z == 1 else "XYZCT"

    # Pixel dtype in metadata
    dtype_str = dtype_meta or np.dtype(tczyx.dtype).name

    return to_ome_arrow(
        image_id=str(image_id or "unnamed"),
        name=str(name or "unknown"),
        image_type=image_type,
        acquisition_datetime=acquisition_datetime or datetime.now(timezone.utc),
        dimension_order=meta_dim_order,
        dtype=dtype_str,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=float(physical_size_x),
        physical_size_y=float(physical_size_y),
        physical_size_z=float(physical_size_z),
        physical_size_unit=str(physical_size_unit),
        channels=channels,
        planes=planes,
        chunk_shape=chunk_shape,
        chunk_order=chunk_order,
        build_chunks=build_chunks,
        masks=None,
    )


def from_tiff(
    tiff_path: str | Path,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    image_type: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
) -> pa.StructScalar:
    """
    Read a TIFF and return a typed OME-Arrow StructScalar.

    Uses bioio to read TCZYX (or XY) data, flattens each YX plane, and
    delegates struct creation to `to_struct_scalar`.

    Args:
        tiff_path: Path to a TIFF readable by bioio.
        image_id: Optional stable image identifier (defaults to stem).
        name: Optional human label (defaults to file name).
        image_type: Optional image kind (e.g., "image", "label").
        channel_names: Optional channel names; defaults to C0..C{n-1}.
        acquisition_datetime: Optional acquisition time (UTC now if None).
        clamp_to_uint16: If True, clamp/cast planes to uint16.

    Returns:
        pa.StructScalar validated against `struct`.
    """

    p = Path(tiff_path)

    img = BioImage(
        image=str(p),
        reader=(
            bioio_ome_tiff.Reader
            if str(p).lower().endswith(("ome.tif", "ome.tiff"))
            else bioio_tifffile.Reader
        ),
    )

    arr = np.asarray(img.data)  # (T, C, Z, Y, X)
    dims = img.dims
    size_t = int(dims.T or 1)
    size_c = int(dims.C or 1)
    size_z = int(dims.Z or 1)
    size_y = int(dims.Y or arr.shape[-2])
    size_x = int(dims.X or arr.shape[-1])
    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dims.")

    psize_x, psize_y, psize_z, unit, _pps_valid = _read_physical_pixel_sizes(img)
    psize_unit = unit or "µm"

    # --- NEW: coerce top-level strings --------------------------------
    img_id = str(image_id or p.stem)
    display_name = str(name or p.name)

    # --- NEW: ensure channel_names is list[str] ------------------------
    if not channel_names or len(channel_names) != size_c:
        channel_names = [f"C{i}" for i in range(size_c)]
    channel_names = [str(x) for x in channel_names]

    channels = [
        {
            "id": f"ch-{i}",
            "name": channel_names[i],
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = arr[t, c, z]
                if clamp_to_uint16 and plane.dtype != np.uint16:
                    plane = np.clip(plane, 0, 65535).astype(np.uint16)
                planes.append(
                    {"z": z, "t": t, "c": c, "pixels": plane.ravel().tolist()}
                )

    dim_order = "XYCT" if size_z == 1 else "XYZCT"

    return to_ome_arrow(
        image_id=img_id,
        name=display_name,
        image_type=image_type,
        acquisition_datetime=acquisition_datetime or datetime.now(timezone.utc),
        dimension_order=dim_order,
        dtype="uint16",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=psize_x,
        physical_size_y=psize_y,
        physical_size_z=psize_z,
        physical_size_unit=psize_unit,
        channels=channels,
        planes=planes,
        masks=None,
    )


def from_stack_pattern_path(
    pattern_path: str | Path,
    default_dim_for_unspecified: str = "C",
    map_series_to: Optional[str] = "T",
    clamp_to_uint16: bool = True,
    channel_names: Optional[List[str]] = None,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    image_type: Optional[str] = None,
) -> pa.StructScalar:
    """Build an OME-Arrow record from a filename pattern describing a stack.

    Args:
        pattern_path: Path or pattern string describing the stack layout.
        default_dim_for_unspecified: Dimension to use when tokens lack a dim.
        map_series_to: Dimension to map series tokens to (e.g., "T"), or None.
        clamp_to_uint16: Whether to clamp pixel values to uint16.
        channel_names: Optional list of channel names to apply.
        image_id: Optional image identifier override.
        name: Optional display name override.
        image_type: Optional image kind (e.g., "image", "label").

    Returns:
        A validated OME-Arrow StructScalar describing the stack.
    """
    path = Path(pattern_path)
    folder = path.parent
    line = path.name.strip()
    if not line:
        raise ValueError("Pattern path string is empty or malformed")

    DIM_TOKENS = {
        "C": {"c", "ch", "w", "wavelength"},
        "T": {"t", "tl", "tp", "timepoint"},
        "Z": {"z", "zs", "sec", "fp", "focal", "focalplane"},
        "S": {"s", "sp", "series"},
    }
    NUM_RANGE_RE = re.compile(r"^(?P<a>\d+)\-(?P<b>\d+)(?::(?P<step>\d+))?$")

    def detect_dim(before_text: str) -> Optional[str]:
        m = re.search(r"([A-Za-z]+)$", before_text)
        if not m:
            return None
        token = m.group(1).lower()
        for dim, names in DIM_TOKENS.items():
            if token in names:
                return dim
        return None

    def expand_raw_token(raw: str) -> Tuple[List[str], bool]:
        raw = raw.strip()
        if "," in raw and not NUM_RANGE_RE.match(raw):
            parts = [p.strip() for p in raw.split(",")]
            return parts, all(p.isdigit() for p in parts)
        m = NUM_RANGE_RE.match(raw)
        if m:
            a, b = m.group("a"), m.group("b")
            step = int(m.group("step") or "1")
            start, stop = int(a), int(b)
            if stop < start:
                raise ValueError(f"Inverted range not supported: <{raw}>")
            width = max(len(a), len(b))
            nums = [str(v).zfill(width) for v in range(start, stop + 1, step)]
            return nums, True
        return [raw], raw.isdigit()

    def parse_bracket_pattern(s: str) -> Tuple[str, List[Dict[str, Any]]]:
        placeholders, out = [], []
        i = ph_i = 0
        while i < len(s):
            if s[i] == "<":
                j = s.find(">", i + 1)
                if j == -1:
                    raise ValueError("Unclosed '<' in pattern.")
                raw_inside = s[i + 1 : j]
                before = "".join(out)
                dim = detect_dim(before) or "?"
                choices, is_num = expand_raw_token(raw_inside)
                placeholders.append(
                    {
                        "idx": ph_i,
                        "raw": raw_inside,
                        "choices": choices,
                        "dim": dim,
                        "is_numeric": is_num,
                    }
                )
                out.append(f"{{{ph_i}}}")
                ph_i += 1
                i = j + 1
            else:
                out.append(s[i])
                i += 1
        return "".join(out), placeholders

    def regex_match(folder: Path, regex: str) -> List[Path]:
        r = re.compile(regex)
        return sorted(
            [p for p in folder.iterdir() if p.is_file() and r.fullmatch(p.name)]
        )

    matched: Dict[Tuple[int, int, int], Path] = {}
    literal_channel_names: Optional[List[str]] = None

    if "<" in line and ">" in line:
        template, placeholders = parse_bracket_pattern(line)
        for ph in placeholders:
            ph["dim"] = (ph["dim"] or "?").upper()
            if ph["dim"] == "?":
                ph["dim"] = default_dim_for_unspecified.upper()

        for combo in itertools.product(*[ph["choices"] for ph in placeholders]):
            fname = template.format(*combo)
            fpath = folder / fname
            if not fpath.exists():
                continue

            t = c = z = 0
            for ph, val in zip(placeholders, combo):
                idx = ph["choices"].index(val)
                dim = ph["dim"]
                if dim == "S":
                    if not map_series_to:
                        raise ValueError("Encountered 'series' but map_series_to=None")
                    dim = map_series_to.upper()
                if dim == "T":
                    t = idx
                elif dim == "C":
                    c = idx
                elif dim == "Z":
                    z = idx

            if literal_channel_names is None:
                for ph in placeholders:
                    dim_eff = ph["dim"] if ph["dim"] != "S" else (map_series_to or "S")
                    if dim_eff == "C" and not ph["is_numeric"]:
                        literal_channel_names = ph["choices"]
                        break

            matched[(t, c, z)] = fpath
    else:
        for z, p in enumerate(regex_match(folder, line)):
            matched[(0, 0, z)] = p

    if not matched:
        raise FileNotFoundError(f"No files matched pattern: {pattern_path}")

    size_t = max(k[0] for k in matched) + 1
    size_c = max(k[1] for k in matched) + 1
    size_z = max(k[2] for k in matched) + 1

    if channel_names and len(channel_names) != size_c:
        raise ValueError(
            f"channel_names length {len(channel_names)} != size_c {size_c}"
        )
    if not channel_names:
        channel_names = literal_channel_names or [f"C{i}" for i in range(size_c)]

    # ---- PROBE SHAPE (NEW: accept TCZYX and squeeze singleton axes) ----
    sample = next(iter(matched.values()))
    is_ome = sample.suffix.lower() in (".ome.tif", ".ome.tiff")
    img0 = BioImage(
        image=str(sample),
        reader=(bioio_ome_tiff.Reader if is_ome else bioio_tifffile.Reader),
    )
    a0 = np.asarray(img0.data)
    # bioio returns TCZYX or YX; normalize to TCZYX
    if a0.ndim == 2:
        _T0, _C0, _Z0, Y0, X0 = 1, 1, 1, a0.shape[0], a0.shape[1]
    else:
        # Heuristic: last two are (Y,X); leading dims are (T,C,Z) possibly singleton
        Y0, X0 = a0.shape[-2], a0.shape[-1]
        lead = a0.shape[:-2]
        # Pad leading dims to T,C,Z (left-aligned)
        _T0, _C0, _Z0 = ([*list(lead), 1, 1, 1])[:3]
    size_y, size_x = Y0, X0

    # physical pixel sizes
    pps = getattr(img0, "physical_pixel_sizes", None)
    try:
        psize_x = float(getattr(pps, "X", None) or 1.0)
        psize_y = float(getattr(pps, "Y", None) or 1.0)
        psize_z = float(getattr(pps, "Z", None) or 1.0)
    except Exception:
        psize_x = psize_y = psize_z = 1.0

    # ---- BUILD PLANES (NEW: support Z-stacks within a single file when T=C=1) ----
    planes: List[Dict[str, Any]] = []

    def _ensure_u16(arr: np.ndarray) -> np.ndarray:
        if clamp_to_uint16 and arr.dtype != np.uint16:
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        return arr

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                fpath = matched.get((t, c, z))
                if fpath is None:
                    # missing plane: zero-fill
                    planes.append(
                        {"z": z, "t": t, "c": c, "pixels": [0] * (size_x * size_y)}
                    )
                    continue

                reader = (
                    bioio_ome_tiff.Reader
                    if fpath.suffix.lower() in (".ome.tif", ".ome.tiff")
                    else bioio_tifffile.Reader
                )
                im = BioImage(image=str(fpath), reader=reader)
                arr = np.asarray(im.data)

                if arr.ndim == 2:
                    # Direct YX
                    if arr.shape != (size_y, size_x):
                        raise ValueError(
                            f"Shape mismatch for {fpath.name}:"
                            f" {arr.shape} vs {(size_y, size_x)}"
                        )
                    arr = _ensure_u16(arr)
                    planes.append(
                        {"z": z, "t": t, "c": c, "pixels": arr.ravel().tolist()}
                    )
                else:
                    # Treat as TCZYX; extract dims
                    Y, X = arr.shape[-2], arr.shape[-1]
                    lead = arr.shape[:-2]
                    Tn, Cn, Zn = ([*list(lead), 1, 1, 1])[:3]
                    if (size_y, size_x) != (Y, X):
                        raise ValueError(
                            f"Shape mismatch for {fpath.name}:"
                            f" {(Y, X)} vs {(size_y, size_x)}"
                        )

                    # Case A: singleton TCZ -> squeeze to YX
                    if Tn == 1 and Cn == 1 and Zn == 1:
                        plane2d = _ensure_u16(arr.reshape(Y, X))
                        planes.append(
                            {"z": z, "t": t, "c": c, "pixels": plane2d.ravel().tolist()}
                        )
                    # Case B: multi-Z only (expand across Z)
                    elif Tn == 1 and Cn == 1 and Zn > 1:
                        # spill Z pages starting at this z index
                        for z_local in range(Zn):
                            plane2d = _ensure_u16(
                                arr.reshape(1, 1, Zn, Y, X)[0, 0, z_local]
                            )
                            z_idx = z + z_local
                            planes.append(
                                {
                                    "z": z_idx,
                                    "t": t,
                                    "c": c,
                                    "pixels": plane2d.ravel().tolist(),
                                }
                            )
                        # bump global size_z if we exceeded it
                        size_z = max(size_z, z + Zn)
                    else:
                        # For now, we require multi-T/C pages to be
                        # expressed by the filename pattern,
                        # not embedded inside a single file.
                        raise ValueError(
                            f"{fpath.name} contains "
                            f"multiple pages across T/C/Z={Tn, Cn, Zn}; "
                            f"only Z>1 with T=C=1 is supported inside one file. "
                            f"Please express T/C via the filename pattern."
                        )

    # Adjust channels (meta)
    channels_meta = [
        {
            "id": f"ch-{i}",
            "name": str((channel_names or [f"C{i}" for i in range(size_c)])[i]),
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    dim_order = "XYZCT" if size_z > 1 else "XYCT"
    display_name = name or str(pattern_path)
    img_id = image_id or path.stem

    return to_ome_arrow(
        image_id=str(img_id),
        name=str(display_name),
        image_type=image_type,
        acquisition_datetime=None,
        dimension_order=dim_order,
        dtype="uint16",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=psize_x,
        physical_size_y=psize_y,
        physical_size_z=psize_z,
        physical_size_unit="µm",
        channels=channels_meta,
        planes=planes,
        masks=None,
    )


def from_ome_zarr(
    zarr_path: str | Path,
    image_id: Optional[str] = None,
    name: Optional[str] = None,
    image_type: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    acquisition_datetime: Optional[datetime] = None,
    clamp_to_uint16: bool = True,
) -> pa.StructScalar:
    """
    Read an OME-Zarr directory and return a typed OME-Arrow StructScalar.

    Uses BioIO with the OMEZarrReader backend to read TCZYX (or XY) data,
    flattens each YX plane into OME-Arrow planes, and builds a validated
    StructScalar via `to_ome_arrow`.

    Args:
        zarr_path:
            Path to the OME-Zarr directory (e.g., "image.ome.zarr").
        image_id:
            Optional stable image identifier (defaults to directory stem).
        name:
            Optional display name (defaults to directory name).
        image_type:
            Optional image kind (e.g., "image", "label").
        channel_names:
            Optional list of channel names. Defaults to C0, C1, ...
        acquisition_datetime:
            Optional datetime (defaults to UTC now).
        clamp_to_uint16:
            If True, cast pixels to uint16.

    Returns:
        pa.StructScalar: Validated OME-Arrow struct for this image.
    """
    p = Path(zarr_path)

    img = BioImage(image=str(p), reader=OMEZarrReader)

    arr = np.asarray(img.data)  # shape (T, C, Z, Y, X)
    dims = img.dims

    size_t = int(dims.T or 1)
    size_c = int(dims.C or 1)
    size_z = int(dims.Z or 1)
    size_y = int(dims.Y or arr.shape[-2])
    size_x = int(dims.X or arr.shape[-1])

    if size_x <= 0 or size_y <= 0:
        raise ValueError("Image must have positive Y and X dimensions.")

    psize_x, psize_y, psize_z, unit, pps_valid = _read_physical_pixel_sizes(img)
    psize_unit = unit or "µm"

    if not pps_valid:
        ngff_scale = _read_ngff_scale(p)
        if ngff_scale is not None:
            psize_x, psize_y, psize_z, unit = ngff_scale
            if unit:
                psize_unit = unit

    img_id = str(image_id or p.stem)
    display_name = str(name or p.name)

    # Infer or assign channel names
    if not channel_names or len(channel_names) != size_c:
        try:
            chs = getattr(img, "channel_names", None)
            if chs is None:
                chs = [getattr(ch, "name", None) for ch in getattr(img, "channels", [])]
            if chs and len(chs) == size_c and all(c is not None for c in chs):
                channel_names = [str(c) for c in chs]
            else:
                channel_names = [f"C{i}" for i in range(size_c)]
        except Exception:
            channel_names = [f"C{i}" for i in range(size_c)]
    channel_names = [str(x) for x in channel_names]

    channels = [
        {
            "id": f"ch-{i}",
            "name": channel_names[i],
            "emission_um": 0.0,
            "excitation_um": 0.0,
            "illumination": "Unknown",
            "color_rgba": 0xFFFFFFFF,
        }
        for i in range(size_c)
    ]

    planes: List[Dict[str, Any]] = []
    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                plane = arr[t, c, z]
                if clamp_to_uint16 and plane.dtype != np.uint16:
                    plane = np.clip(plane, 0, 65535).astype(np.uint16)
                planes.append(
                    {"z": z, "t": t, "c": c, "pixels": plane.ravel().tolist()}
                )

    dim_order = "XYCT" if size_z == 1 else "XYZCT"

    return to_ome_arrow(
        image_id=img_id,
        name=display_name,
        image_type=image_type,
        acquisition_datetime=acquisition_datetime or datetime.now(timezone.utc),
        dimension_order=dim_order,
        dtype="uint16",
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        size_c=size_c,
        size_t=size_t,
        physical_size_x=psize_x,
        physical_size_y=psize_y,
        physical_size_z=psize_z,
        physical_size_unit=psize_unit,
        channels=channels,
        planes=planes,
        masks=None,
    )


def from_ome_parquet(
    parquet_path: str | Path,
    *,
    column_name: Optional[str] = "ome_arrow",
    row_index: int = 0,
    strict_schema: bool = False,
) -> pa.StructScalar:
    """Read an OME-Arrow record from a Parquet file.

    Args:
        parquet_path: Path to the Parquet file.
        column_name: Column to read; auto-detected when None or invalid.
        row_index: Row index to extract.
        strict_schema: Require the exact OME-Arrow schema if True.

    Returns:
        A typed OME-Arrow StructScalar.

    Raises:
        FileNotFoundError: If the Parquet path does not exist.
        ValueError: If the row index is out of range or no suitable column exists.
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    table = pq.read_table(p)
    return _ome_arrow_from_table(
        table,
        column_name=column_name,
        row_index=row_index,
        strict_schema=strict_schema,
    )


def from_ome_vortex(
    vortex_path: str | Path,
    *,
    column_name: Optional[str] = "ome_arrow",
    row_index: int = 0,
    strict_schema: bool = False,
) -> pa.StructScalar:
    """Read an OME-Arrow record from a Vortex file.

    Args:
        vortex_path: Path to the Vortex file.
        column_name: Column to read; auto-detected when None or invalid.
        row_index: Row index to extract.
        strict_schema: Require the exact OME-Arrow schema if True.

    Returns:
        A typed OME-Arrow StructScalar.

    Raises:
        FileNotFoundError: If the Vortex path does not exist.
        ImportError: If the optional `vortex-data` dependency is missing.
        ValueError: If the row index is out of range or no suitable column exists.
    """
    p = Path(vortex_path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    try:
        import vortex
    except ImportError as exc:
        raise ImportError(
            "Vortex support requires the optional 'vortex-data' dependency."
        ) from exc

    table = vortex.open(str(p)).to_arrow().read_all()
    return _ome_arrow_from_table(
        table,
        column_name=column_name,
        row_index=row_index,
        strict_schema=strict_schema,
    )
