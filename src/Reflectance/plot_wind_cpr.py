"""
Step 3 — Plot SWIR2 reflectance with wind vectors and CPR ground track.

Reads masked reflectance NetCDF files produced by mask_reflectance.py,
fetches the corresponding AUX_MET_1D (wind) and ACM_CAP_2B (CPR track)
products from MAAP, and produces one map per orbit.

Wind arrows come from AUX_MET_1D at model level 96 (≈ 950 hPa).
The CPR ground track is overlaid as a cyan line.

Output
------
<project>/plots/wind_cpr/<orbit>_<frame>_<datetime>.png
<project>/tracks/plot_wind_cpr_<run_ts>.log
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from auth import get_token
from maap_loader import open_catalog, search_products, make_filesystem, load_hdf5_dataset

# ── configuration ─────────────────────────────────────────────────────────────

# Point of interest — used only for the POI marker; extent comes from the data.
POI_LAT = 41.0
POI_LON = -87.5

# Model level for wind arrows (AUX_MET_1D uses ECMWF 137-level scheme;
# level 96 in the stored indexing corresponds to roughly 950 hPa).
WIND_LEVEL = 137 - 96

# Directory produced by mask_reflectance.py
DATA_DIR = Path(__file__).parent.parent / "data" / "reflectance_masked"

PLOT_DIR = Path(__file__).parent.parent / "plots" / "wind_cpr"
LOG_DIR  = Path(__file__).parent.parent / "tracks"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_run_ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_log_path = LOG_DIR / f"plot_wind_cpr_{_run_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("Run started — log: %s", _log_path)

# ── catalog connection ─────────────────────────────────────────────────────────

token   = get_token()
catalog = open_catalog()
fs      = make_filesystem(token)

# ── helpers ────────────────────────────────────────────────────────────────────

def _roi_filter(lat, lon, lat_range, lon_range):
    return (
        np.isfinite(lat) & np.isfinite(lon) &
        (lat >= lat_range[0]) & (lat <= lat_range[1]) &
        (lon >= lon_range[0]) & (lon <= lon_range[1])
    )


def _fetch_cpr_track(catalog, orbit, frame, lat_range, lon_range):
    """Return (lat, lon) arrays for CPR track within the ROI, or (None, None)."""
    hits = search_products(
        catalog,
        collection="EarthCAREL2Validated_MAAP",
        filter_expr=(
            f"productType = 'ACM_CAP_2B' "
            f"and orbitNumber = {orbit} and frame = '{frame}'"
        ),
        max_items=1,
    )
    if not hits:
        return None, None
    ds   = load_hdf5_dataset(fs, hits[0], group="ScienceData")
    lat  = ds["latitude"].where(ds["latitude"] < 1e36).values.ravel()
    lon  = ds["longitude"].where(ds["longitude"] < 1e36).values.ravel()
    mask = _roi_filter(lat, lon, lat_range, lon_range)
    return lat[mask], lon[mask]


def _fetch_wind(catalog, orbit, frame, lat_range, lon_range):
    """Return (lat, lon, u, v) arrays for wind within the ROI, or (None,)*4."""
    hits = search_products(
        catalog,
        collection="EarthCAREXMETL1DProducts10_MAAP",
        filter_expr=(
            f"productType = 'AUX_MET_1D' "
            f"and track = '{orbit:05d}' "
            f"and frame = '{frame}' "
            f"and version = 'BC'"
        ),
        max_items=1,
    )
    if not hits:
        return None, None, None, None
    ds      = load_hdf5_dataset(fs, hits[0], group="ScienceData")
    u       = ds["eastward_wind"].sel(height=WIND_LEVEL).values.ravel()
    v       = ds["northward_wind"].sel(height=WIND_LEVEL).values.ravel()
    lat_aux = ds["latitude"].values.ravel()
    lon_aux = ds["longitude"].values.ravel()
    valid   = (
        _roi_filter(lat_aux, lon_aux, lat_range, lon_range) &
        np.isfinite(u) & np.isfinite(v)
    )
    return lat_aux[valid], lon_aux[valid], u[valid], v[valid]


# ── main loop ─────────────────────────────────────────────────────────────────

nc_files = sorted(DATA_DIR.glob("SWIR2_masked_*.nc"))
log.info("Masked reflectance files found: %d", len(nc_files))

for nc_path in nc_files:
    # Filename: SWIR2_masked_<orbit>_<frame>_<dt>.nc
    parts  = nc_path.stem.split("_")       # ['SWIR2', 'masked', orbit, frame, dt]
    orbit  = int(parts[2])
    frame  = parts[3]
    dt_str = parts[4]

    out_path = PLOT_DIR / f"wind_cpr_{orbit}_{frame}_{dt_str}.png"
    if out_path.exists():
        log.info("Skip (exists): %s", out_path.name)
        continue

    log.info("Plotting orbit=%s  frame=%s  dt=%s", orbit, frame, dt_str)
    try:
        ds = xr.open_dataset(nc_path)
        lat = ds["lat"].values
        lon = ds["lon"].values
        ref = ds["reflectance"].values

        # Derive plot extent from the data
        lat_range = (float(lat.min()), float(lat.max()))
        lon_range = (float(lon.min()), float(lon.max()))
        extent    = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]

        log.info("  %d reflectance points  lat=%.2f–%.2f  lon=%.2f–%.2f",
                 len(lat), *lat_range, *lon_range)

        # Fetch auxiliary data
        cpr_lat, cpr_lon             = _fetch_cpr_track(catalog, orbit, frame, lat_range, lon_range)
        wind_lat, wind_lon, u_wind, v_wind = _fetch_wind(catalog, orbit, frame, lat_range, lon_range)

        log.info("  CPR track points : %s", len(cpr_lat) if cpr_lat is not None else "N/A")
        log.info("  Wind points      : %s", len(wind_lat) if wind_lat is not None else "N/A")

        # ── figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(
            figsize=(10, 7),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="red")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="orange")
        ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="gray")

        # Reflectance scatter
        sc = ax.scatter(
            lon, lat, c=ref,
            cmap="turbo", vmin=0.0, vmax=0.5,
            s=2, transform=ccrs.PlateCarree(), zorder=4,
        )
        plt.colorbar(sc, ax=ax, label="TOA Reflectance (SWIR2)", shrink=0.7)

        # CPR track
        if cpr_lat is not None and len(cpr_lat) > 0:
            ax.plot(
                cpr_lon, cpr_lat,
                color="cyan", linewidth=1.5,
                transform=ccrs.PlateCarree(), zorder=5, label="CPR track",
            )

        # Wind arrows
        if wind_lat is not None and len(wind_lat) > 0:
            ax.quiver(
                wind_lon, wind_lat, u_wind, v_wind,
                color="gray", scale=300, width=0.004, alpha=0.5,
                transform=ccrs.PlateCarree(), zorder=6, label=f"Wind lvl {WIND_LEVEL}",
            )

        # POI marker
        ax.plot(
            POI_LON, POI_LAT,
            marker="x", color="red", markersize=10, markeredgewidth=2,
            transform=ccrs.PlateCarree(), zorder=7, label="POI",
        )

        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(
            f"SWIR2 Reflectance + Wind + CPR — orbit {orbit}  frame {frame}  {dt_str}"
        )
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", out_path.name)

        ds.close()

    except Exception as exc:
        log.error("Failed orbit=%s frame=%s: %s", orbit, frame, exc)
        plt.close("all")

log.info("Done. Output directory: %s", PLOT_DIR)
