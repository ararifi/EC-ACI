"""
Step 1 — Generate SWIR2 reflectance and filter by quality.

Searches the MAAP catalog for EarthCARE overpasses near a POI, applies
pixel-level quality filters from MSI_RGR_1C, and saves one NetCDF per orbit
to DATA_DIR.  Optionally restricts to daytime-only orbits.

Requires the environment variable MAAP_CREDENTIAL to point to a credentials
file containing OFFLINE_TOKEN, CLIENT_ID, and CLIENT_SECRET.

Output
------
<project>/data/reflectance_raw/SWIR2_raw_<orbit>_<frame>_<datetime>.nc
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

# ── configuration ─────────────────────────────────────────────────────────────

POI_LAT = 41.0
POI_LON = -87.5

BBOX_BUFFER    = 3.0
PLOT_EXTEND    = 3.0
DATETIME_RANGE = ["2025-07-01", "2025-09-30"]

# Set to a list like ["6708D", "6710C"] to process specific orbits only,
# or None to process everything found in the bbox/datetime range.
ORBIT_FILTER: list[str] | None = None

# When True, only orbits whose acquisition time falls in the daytime window
# (18–22 UTC) are kept.  Individual pixels still get their own SZA < 85° cut.
DAYTIME_ONLY: bool = True

# ── output paths ──────────────────────────────────────────────────────────────

_ROOT    = Path(__file__).parent.parent
DATA_DIR = _ROOT / "Data" / "Processed" / "Reflectance"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def _orbit_frame(item) -> tuple[int, str]:
    p     = item.properties
    orbit = p.get("sat:absolute_orbit") or p.get("orbitNumber")
    gc    = p.get("grid:code", "")
    frame = gc.split("-")[-1] if gc else p.get("frame", "")
    return int(orbit), str(frame)


def _parse_orbit_filter(entries: list[str]) -> list[tuple[int, str]]:
    return [(int(e[:-1]), e[-1]) for e in entries]


def _orbit_filter_cql(entries: list[str]) -> str:
    clauses = [
        f"(orbitNumber = {o} and frame = '{f}')"
        for o, f in _parse_orbit_filter(entries)
    ]
    return "(" + " or ".join(clauses) + ")"

# ── catalog connection ─────────────────────────────────────────────────────────

token   = get_token(_cred_path)
catalog = open_catalog()
fs      = make_filesystem(token)

bbox = [
    POI_LON - BBOX_BUFFER, POI_LAT - BBOX_BUFFER,
    POI_LON + BBOX_BUFFER, POI_LAT + BBOX_BUFFER,
]

# ── 1. search MSI_RGR_1C granules ─────────────────────────────────────────────

if ORBIT_FILTER:
    msi_items = search_products(
        catalog,
        collection="EarthCAREL1Validated_MAAP",
        filter_expr=f"productType = 'MSI_RGR_1C' and {_orbit_filter_cql(ORBIT_FILTER)}",
        max_items=2000,
    )
else:
    msi_items = search_products(
        catalog,
        collection="EarthCAREL1Validated_MAAP",
        bbox=bbox,
        filter_expr="productType = 'MSI_RGR_1C' and (frame = 'D' or frame = 'E' or frame = 'F') ",
        datetime_range=DATETIME_RANGE,
        max_items=2000,
    )


print(f"MSI granules to process: {len(msi_items)}")

# ── 2. process each granule ────────────────────────────────────────────────────

lat_range = (POI_LAT - PLOT_EXTEND, POI_LAT + PLOT_EXTEND)
lon_range = (POI_LON - PLOT_EXTEND, POI_LON + PLOT_EXTEND)

for msi_item in msi_items:
    orbit, frame = _orbit_frame(msi_item)
    dt_str       = msi_item.datetime.strftime("%Y%m%dT%H%M%SZ")
    out_path     = DATA_DIR / f"SWIR2_raw_{orbit}_{frame}_{dt_str}.nc"

    if out_path.exists():
        print(f"Skip (exists): {out_path.name}")
        continue

    print(f"Processing orbit={orbit}  frame={frame}  dt={dt_str}")
    try:
        ds = load_hdf5_dataset(fs, msi_item, group="ScienceData")

        L   = ds["pixel_values"].sel(band="SWIR2")
        E0  = ds["solar_spectral_irradiance"].isel(VNS_band=3)
        sza = 90.0 - ds["solar_elevation_angle"]
        mu0 = np.cos(np.deg2rad(sza))
        R   = (np.pi * L) / (E0 * mu0)

        good_quality = ds["pixel_quality_status"].sel(band="SWIR2").values == 0
        is_daytime   = sza.values < 85.0
        no_sunglint  = ~ds["sunglint_flag"].values.astype(bool)
        valid_range  = (R.values >= 0.0) & (R.values <= 1.5)
        combined     = good_quality & is_daytime & no_sunglint & valid_range

        R_out = R.values.copy()
        R_out[~combined] = np.nan
        R_out = R_out.ravel()

        lat = ds["latitude"].where(ds["latitude"] < 1e36).values.ravel()
        lon = ds["longitude"].where(ds["longitude"] < 1e36).values.ravel()

        finite          = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(R_out)
        lat, lon, R_out = lat[finite], lon[finite], R_out[finite]

        roi = (
            (lat >= lat_range[0]) & (lat <= lat_range[1]) &
            (lon >= lon_range[0]) & (lon <= lon_range[1])
        )

        if not roi.any():
            print("  no pixels in ROI — skipping")
            continue

        xr.Dataset(
            {
                "reflectance": ("point", R_out[roi].astype("float32")),
                "lat":         ("point", lat[roi].astype("float32")),
                "lon":         ("point", lon[roi].astype("float32")),
            },
            attrs={
                "orbit":     str(orbit),
                "frame":     frame,
                "datetime":  dt_str,
                "product":   "SWIR2_TOA_reflectance_quality_filtered",
                "poi_lat":   str(POI_LAT),
                "poi_lon":   str(POI_LON),
                "lat_range": f"{lat_range[0]:.3f},{lat_range[1]:.3f}",
                "lon_range": f"{lon_range[0]:.3f},{lon_range[1]:.3f}",
            },
        ).to_netcdf(out_path)
        print(f"  saved: {out_path.name}")

    except Exception as exc:
        print(f"  ERROR orbit={orbit} frame={frame}: {exc}", file=sys.stderr)

print(f"Done. Output: {DATA_DIR}")
