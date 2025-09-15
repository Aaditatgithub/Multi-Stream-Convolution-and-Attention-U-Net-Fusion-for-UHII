# Export UHI / LST / PM2.5 tensors (monthly) — explanation & usage


## Procedure (step-by-step)

1. **Initializes** Google Earth Engine (auth if needed) using the specified `cloud_project`.
2. **Defines** city center locations and creates `GRID_SIZE_KM` rectangular regions (40 km × 40 km) in geographic degrees.
3. **Iterates** months over `YEARS = 2003..2020` and `MONTHS = 1..12` (T = 18 years × 12 months = 216 time steps).
4. **Downloads / computes** for each city & month:

   * **PM2.5** from `GLOBAL-SATELLITE-PM25/MONTHLY` → resampled to 1 km grid (40×40 pixels) → stored as `(H,W)` per time step with availability mask.
   * **LST (MODIS)** monthly mean from `MOD11A2` → two bands `LST_Day`, `LST_Night` (°C) → resampled to 1 km grid → stored as `(2,H,W)` per time step with masks.
   * **UHII**: multiple prebuilt UHII collections (order in `UHII_BANDS`) — each band is meaned for the month, masked for values > 0, multiplied by 0.01 (°C), resampled to 1 km → stored as `(B,H,W)` per time step with masks.
5. **Downloads annual land cover** from `GLC-FCS30D/annual` at 30 m resolution for each year in `YEARS`, recodes classes and stores yearly tiles at \~1333×1333 (30 m) per year.
6. **Creates availability masks** (simple `> 0` criteria for UHII/PM/LST — LST uses nonzero check).
7. **Saves** tensors and a `meta.json` per city in `/content/UHI_Tensors/{city}`.

---

## Key variables & outputs

* Time steps: `T = 216` (2003–2020 monthly).
* UHII bands (order preserved): `UHII_BANDS = ["AMOD2","MOD1","MOD2","MYD1","MYD2","SAT","SMOD2","SMYD1"]` (B = 8).
* Grid sizes:

  * 1 km grid: `WIDTH_PX_1KM = 40` → 40 × 40 pixels (this covers 40 km with 1 km pixels).
  * 30 m annual LC grid: `WIDTH_PX_30M = 1333` → \~1333 × 1333 pixels (≈ 40 km at 30 m).
* Saved files per city (in `/content/UHI_Tensors/{city}`):

  * `pm25.pt` — shape `(T, 1, 40, 40)` (torch tensor)
  * `pm25_mask.pt` — `(T, 1, 40, 40)`
  * `lst.pt` — `(T, 2, 40, 40)` (Day, Night)
  * `lst_mask.pt` — `(T, 2, 40, 40)`
  * `uhii.pt` — `(T, B, 40, 40)`
  * `uhii_mask.pt` — `(T, B, 40, 40)`
  * `lc_annual.pt` — `(Y, 1, 1333, 1333)` where `Y = number of years (18)`
  * `meta.json` — contains `years`, `months`, `t_steps`, grid sizes, band order, and file map.

---

## Important functions (quick reference)

* `create_region_geometry(lon, lat, size_km)` — convert city center to approximately square geographic region.
* `to1km(img, region)` — reprojects and re-scales an `ee.Image` to 1 km (EPSG:4326).
* `ee_to_numpy_singleband(...)` / `ee_to_numpy_multiband(...)` — download via `geemap.ee_to_numpy`, handle missing values, ensure consistent target shapes, and use `scipy.ndimage.zoom` if resizing is necessary.
* `fetch_lst_month(year, month, region)` — fetches monthly LST and converts Kelvin → °C.
* `recode_classes(img)` & `lc_band_for_year(year, region)` — prepare annual landcover image from the GLC-FCS30D asset.
* `availability_mask_from_array(arr, thresh=0.0)` — returns float mask (1.0 valid, 0.0 invalid) using `> thresh`.

---

## Assumptions, design notes & caveats

* **Resampling / projection**: `reproject(..., scale=PIXEL_SIZE_1KM)` uses EPSG:4326 which can produce approximate pixel sizes in degrees — acceptable for moderate regional analysis but not perfect for precise area calculations near high latitudes. For India this is usually fine.
* **Masking criterion**: validity is determined by simple `> 0` or `!= 0` checks. Depending on the dataset, you might prefer dataset-specific quality flags or NaN handling.
* **geemap behavior**: `geemap.ee_to_numpy` may return `None` if the request fails or the tile is empty; the code substitutes zeros in that case. You may prefer to retry or raise errors instead.
* **Memory & time**: downloading 216 monthly time steps × several bands × multiple cities and a 1333×1333 × 18 annual array is heavy. Expect long runtimes and substantial memory / disk usage. Run on a machine with enough RAM and disk space; consider processing fewer years or smaller extents for testing.
* **Units**:

  * MODIS LST is converted to °C via `.multiply(0.02).subtract(273.15)`.
  * UHII assets are multiplied by `0.01` (as in the code).
* **`LC` recoding**: the `recode_classes` remap mapping is present but may need adjustment depending on the target class groups you want.

---

## Troubleshooting tips

* If Earth Engine raises authentication errors, run `ee.Authenticate()` (the try/except already does this once).
* If `geemap.ee_to_numpy` times out or returns `None`, try smaller regions / coarser scale to debug, or add retries and exponential backoff.
* If memory spikes when stacking tensors, consider saving monthly slices incrementally or using `torch.save` per-time-step and merging later.
* If shapes mismatch, the helper `resize_array_to_target` uses `scipy.ndimage.zoom(order=1)` to ensure consistent shapes; check `target_shape` arguments if you want different resolution.
