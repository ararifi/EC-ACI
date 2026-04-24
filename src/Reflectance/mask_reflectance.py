"""
Step 2 — Apply cloud-property masks to quality-filtered reflectance.

Reads NetCDF files produced by generate_reflectance.py, fetches the
corresponding MSI_CM__2A and MSI_COP_2A products from MAAP, applies
cloud-property masks, and writes masked NetCDF files to DATA_DIR_OUT.

Cloud mask criteria (all must be satisfied):
  - quality_status <= 2               — valid pixel (not invalid/twilight)
  - cloud_mask in {2, 3}              — probably or confidently cloudy
  - cloud_phase == 1                  — liquid-water cloud
  - cloud_type in {1, 2, 4, 5, 7, 8} — low/mid-altitude cloud types
  - cloud_top_pressure > 50000 Pa     — cloud top below the 500 hPa level

Requires the environment variable MAAP_CREDENTIAL to point to a credentials
file containing OFFLINE_TOKEN, CLIENT_ID, and CLIENT_SECRET.

Output
------
<project>/data/reflectance_masked/SWIR2_masked_<orbit>_<frame>_<datetime>.nc
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from auth import get_token
from maap_loader import open_catalog, search_products, make_filesystem, load_hdf5_dataset

# ── credentials ───────────────────────────────────────────────────────────────

_cred_path = os.environ.get("MAAP_CREDENTIAL")
if not _cred_path:
    raise EnvironmentError("MAAP_CREDENTIAL environment variable is not set.")

# ── paths ─────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).parent.parent
DATA_DIR_IN  = _ROOT / "data" / "reflectance_raw"
DATA_DIR_OUT = _ROOT / "data" / "reflectance_masked"
DATA_DIR_OUT.mkdir(parents=True, exist_ok=True)

# ── catalog connection ─────────────────────────────────────────────────────────

token   = get_token(_cred_path)
catalog = open_catalog()
fs      = make_filesystem(token)

# ── helpers ────────────────────────────────────────────────────────────────────

def _build_cloud_mask(ds_cm: xr.Dataset, ds_cop: xr.Dataset) -> np.ndarray:
    """Return a boolean pixel mask: True where all cloud criteria are met."""
    good_quality   = ds_cm["quality_status"].values <= 2
    is_cloudy      = (ds_cm["cloud_mask"].values >= 2) & (ds_cm["cloud_mask"].values <= 3)
    is_water_phase = ds_cm["cloud_phase"].values == 1

    ct         = ds_cm["cloud_type"].values
    is_low_mid = (ct == 1) | (ct == 2) | (ct == 4) | (ct == 5) | (ct == 7) | (ct == 8)

    ctp          = ds_cop["cloud_top_pressure"].values
    is_low_cloud = (ctp > 50_000.0) & np.isfinite(ctp)

    return good_quality & is_cloudy & is_water_phase & is_low_mid & is_low_cloud


# ── process each raw reflectance file ─────────────────────────────────────────

raw_files = sorted(DATA_DIR_IN.glob("SWIR2_raw_*.nc"))
print(f"Raw reflectance files found: {len(raw_files)}")

for nc_in in raw_files:
    # Filename: SWIR2_raw_<orbit>_<frame>_<dt>.nc
    parts  = nc_in.stem.split("_")   # ['SWIR2', 'raw', orbit, frame, dt]
    orbit  = int(parts[2])
    frame  = parts[3]
    dt_str = parts[4]

    nc_out = DATA_DIR_OUT / f"SWIR2_masked_{orbit}_{frame}_{dt_str}.nc"
    if nc_out.exists():
        print(f"Skip (exists): {nc_out.name}")
        continue

    print(f"Processing orbit={orbit}  frame={frame}  dt={dt_str}")

    cm_hits = search_products(
        catalog,
        collection="EarthCAREL2Validated_MAAP",
        filter_expr=(
            f"productType = 'MSI_CM__2A' and version = 'BA' "
            f"and orbitNumber = {orbit} and frame = '{frame}'"
        ),
    )
    cop_hits = search_products(
        catalog,
        collection="EarthCAREL2Validated_MAAP",
        filter_expr=(
            f"productType = 'MSI_COP_2A' "
            f"and orbitNumber = {orbit} and frame = '{frame}'"
        ),
    )

    if not cm_hits:
        print("  MSI_CM__2A not found — skipping", file=sys.stderr)
        continue
    if not cop_hits:
        print("  MSI_COP_2A not found — skipping", file=sys.stderr)
        continue

    try:
        ds_raw = xr.open_dataset(nc_in)
        ds_cm  = load_hdf5_dataset(fs, cm_hits[0],  group="ScienceData")
        ds_cop = load_hdf5_dataset(fs, cop_hits[0], group="ScienceData")

        cloud_mask_flat = _build_cloud_mask(ds_cm, ds_cop).ravel()

        lat_raw = ds_raw["lat"].values
        lon_raw = ds_raw["lon"].values

        # Align by nearest neighbour between saved ROI points and the CM grid
        lat_cm = ds_cm["latitude"].where(ds_cm["latitude"] < 1e36).values.ravel()
        lon_cm = ds_cm["longitude"].where(ds_cm["longitude"] < 1e36).values.ravel()

        keep = np.zeros(len(lat_raw), dtype=bool)
        for i, (la, lo) in enumerate(zip(lat_raw, lon_raw)):
            idx     = int(np.argmin((lat_cm - la) ** 2 + (lon_cm - lo) ** 2))
            keep[i] = cloud_mask_flat[idx] if idx < len(cloud_mask_flat) else False

        print(f"  pixels passing cloud mask: {keep.sum()} / {len(keep)}")

        if not keep.any():
            print("  no valid pixels — skipping")
            ds_raw.close()
            continue

        xr.Dataset(
            {
                "reflectance": ("point", ds_raw["reflectance"].values[keep]),
                "lat":         ("point", lat_raw[keep]),
                "lon":         ("point", lon_raw[keep]),
            },
            attrs={
                **ds_raw.attrs,
                "product":       "SWIR2_TOA_reflectance_cloud_masked",
                "cloud_filters": (
                    "quality_status<=2; cloud_mask in {2,3}; "
                    "cloud_phase==1; cloud_type in {1,2,4,5,7,8}; "
                    "cloud_top_pressure>50000Pa"
                ),
            },
        ).to_netcdf(nc_out)
        ds_raw.close()
        print(f"  saved: {nc_out.name}")

    except Exception as exc:
        print(f"  ERROR orbit={orbit} frame={frame}: {exc}", file=sys.stderr)

print(f"Done. Output: {DATA_DIR_OUT}")