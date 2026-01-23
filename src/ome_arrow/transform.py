"""
Module for transforming OME-Arrow data
(e.g., slices, projections, or other changes).
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from ome_arrow.ingest import _build_chunks_from_planes, _normalize_chunk_shape
from ome_arrow.meta import OME_ARROW_STRUCT


def slice_ome_arrow(
    data: Dict[str, Any] | pa.StructScalar,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    t_indices: Optional[Iterable[int]] = None,
    c_indices: Optional[Iterable[int]] = None,
    z_indices: Optional[Iterable[int]] = None,
    fill_missing: bool = True,
) -> pa.StructScalar:
    """
    Create a cropped copy of an OME-Arrow record.

    Crops spatially to [y_min:y_max, x_min:x_max] (half-open) and, if provided,
    filters/reindexes T/C/Z to the given index sets.

    Parameters
    ----------
    data : dict | pa.StructScalar
        OME-Arrow record.
    x_min, x_max, y_min, y_max : int
        Half-open crop bounds in pixels (0-based).
    t_indices, c_indices, z_indices : Iterable[int] | None
        Optional explicit indices to keep for T, C, Z. If None, keep all.
        Selected indices are reindexed to 0..len-1 in the output.
    fill_missing : bool
        If True, any missing (t,c,z) planes in the selection are zero-filled.

    Returns
    -------
    pa.StructScalar
        New OME-Arrow record with updated sizes and planes.
    """
    # Unwrap to dict
    row = data.as_py() if isinstance(data, pa.StructScalar) else dict(data)
    pm = dict(row.get("pixels_meta", {}))

    sx = int(pm.get("size_x", 1))
    sy = int(pm.get("size_y", 1))
    sz = int(pm.get("size_z", 1))
    sc = int(pm.get("size_c", 1))
    st = int(pm.get("size_t", 1))
    if not (0 <= x_min < x_max <= sx and 0 <= y_min < y_max <= sy):
        raise ValueError(
            f"Crop bounds out of range: x[{x_min},{x_max}) within [0,{sx}), "
            f"y[{y_min},{y_max}) within [0,{sy})."
        )

    # Normalize T/C/Z selections (keep all if None)
    def _norm(sel: Optional[Iterable[int]], size: int) -> List[int]:
        return (
            list(range(size))
            if sel is None
            else sorted({int(i) for i in sel if 0 <= int(i) < size})
        )

    keep_t = _norm(t_indices, st)
    keep_c = _norm(c_indices, sc)
    keep_z = _norm(z_indices, sz)
    if len(keep_t) == 0 or len(keep_c) == 0 or len(keep_z) == 0:
        raise ValueError("Selection must keep at least one index in each of T/C/Z.")

    # Reindex maps (old -> new)
    t_map = {t: i for i, t in enumerate(keep_t)}
    c_map = {c: i for i, c in enumerate(keep_c)}
    z_map = {z: i for i, z in enumerate(keep_z)}

    new_sx = x_max - x_min
    new_sy = y_max - y_min
    new_st = len(keep_t)
    new_sc = len(keep_c)
    new_sz = len(keep_z)

    # Fast access to incoming planes
    planes_in: List[Dict[str, Any]] = list(row.get("planes", []))
    if not planes_in:
        raise ValueError("Record contains no planes to slice.")

    # Group incoming planes by (t,c,z)
    by_tcz: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for p in planes_in:
        tt = int(p["t"])
        cc = int(p["c"])
        zz = int(p["z"])
        by_tcz[(tt, cc, zz)] = p

    # Helper to crop one plane
    expected_len = sx * sy

    def _crop_pixels(flat: Iterable[int]) -> List[int]:
        arr = np.asarray(flat)
        if arr.size != expected_len:
            # be strict: malformed plane
            raise ValueError(f"Plane has {arr.size} pixels; expected {expected_len}.")
        arr = arr.reshape(sy, sx)
        sub = arr[y_min:y_max, x_min:x_max]
        return sub.ravel().astype(arr.dtype, copy=False).tolist()

    # Build new plane list in dense (t,c,z) order using selections
    planes_out: List[Dict[str, Any]] = []
    for tt in keep_t:
        for cc in keep_c:
            for zz in keep_z:
                src = by_tcz.get((tt, cc, zz))
                if src is None:
                    if not fill_missing:
                        continue
                    # zero-fill missing plane
                    planes_out.append(
                        {
                            "t": t_map[tt],
                            "c": c_map[cc],
                            "z": z_map[zz],
                            "pixels": [0] * (new_sx * new_sy),
                        }
                    )
                else:
                    cropped = _crop_pixels(src["pixels"])
                    planes_out.append(
                        {
                            "t": t_map[tt],
                            "c": c_map[cc],
                            "z": z_map[zz],
                            "pixels": cropped,
                        }
                    )

    # Filter channel metadata to kept channels and reindex
    channels_in = list(pm.get("channels", []) or [])
    channels_out: List[Dict[str, Any]] = []
    # If channels metadata length mismatches, synthesize minimal entries
    if len(channels_in) != sc:
        channels_in = [
            {"id": f"ch-{i}", "name": f"C{i}", "color_rgba": 0xFFFFFFFF}
            for i in range(sc)
        ]
    for old_c in keep_c:
        meta = dict(channels_in[old_c])
        meta["id"] = f"ch-{c_map[old_c]}"
        # ensure name string
        if "name" in meta:
            meta["name"] = str(meta["name"])
        else:
            meta["name"] = f"C{c_map[old_c]}"
        channels_out.append(meta)

    # Update pixels_meta
    pm_out = dict(pm)
    pm_out.update(
        {
            "size_x": new_sx,
            "size_y": new_sy,
            "size_z": new_sz,
            "size_c": new_sc,
            "size_t": new_st,
            "channels": channels_out,
        }
    )

    # If dimension order encoded XYCT/XYZCT etc., keep it as-is (no axis permutation).
    # (Optional: you could normalize to XYCT if new_sz==1, else XYZCT.)

    # Assemble new record
    rec_out = dict(row)
    rec_out["pixels_meta"] = pm_out
    rec_out["planes"] = planes_out

    chunk_grid_in = row.get("chunk_grid") or {}
    if chunk_grid_in or row.get("chunks"):
        chunk_shape = (
            int(chunk_grid_in.get("chunk_z", 1)),
            int(chunk_grid_in.get("chunk_y", 512)),
            int(chunk_grid_in.get("chunk_x", 512)),
        )
        chunk_order = str(chunk_grid_in.get("chunk_order") or "ZYX")
        chunks_out = _build_chunks_from_planes(
            planes=planes_out,
            size_t=new_st,
            size_c=new_sc,
            size_z=new_sz,
            size_y=new_sy,
            size_x=new_sx,
            chunk_shape=chunk_shape,
            chunk_order=chunk_order,
        )
        cz, cy, cx = _normalize_chunk_shape(chunk_shape, new_sz, new_sy, new_sx)
        rec_out["chunk_grid"] = {
            "order": "TCZYX",
            "chunk_t": 1,
            "chunk_c": 1,
            "chunk_z": cz,
            "chunk_y": cy,
            "chunk_x": cx,
            "chunk_order": chunk_order,
        }
        rec_out["chunks"] = chunks_out
    else:
        rec_out["chunk_grid"] = row.get("chunk_grid")
        rec_out["chunks"] = row.get("chunks")

    return pa.scalar(rec_out, type=OME_ARROW_STRUCT)
