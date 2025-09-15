# Finalize & prepare tensors — explanation & usage

**One-line summary:**
This Colab cell loads per-city exported tensors and metadata, validates shapes, expands the annual 30 m land-cover tiles into a monthly sequence aligned with the monthly time axis (T), saves the new `lc.pt` and a `month_to_year.pt` index, and updates `meta.json`.

---

## What the script does (step-by-step)

1. Optionally **unzips** `UHI_Tensors.zip` into a workspace (via `extract_zip_if_needed`) and locates the city subfolders.
2. For each city folder found and expected (Delhi, Mumbai, …), it:

   * Loads `meta.json` to get expected shapes / years / months / band order.
   * Loads tensors using `_load()` (which reads files from `meta["files"]` or falls back to `{name}.pt`).
   * Validates the shapes of PM2.5, LST, UHII tensors and masks against the expected `(T, C, H, W)` shapes.
   * Loads `lc_annual` (expected shape `(Y, 1, H30, W30)`) and computes `Y`.
3. Builds a `month_to_year` index array of length `T` mapping each monthly time index `t` → `year_index = min(Y-1, t // 12)`.
4. **Replicates annual LC to monthly** with `lc_annual.repeat_interleave(12, dim=0)[:T]` producing `lc_monthly` shaped `(T, 1, H30, W30)`.
5. Saves `lc.pt` (the monthly LC), `month_to_year.pt`, and updates `meta.json` with file names and a short note.
6. Prints shape summaries if `VERBOSE` is `True`.

---

## Key variables & outputs

* `T` — total time steps (from `meta["t_steps"]`). Default computed as `len(YEARS) * len(MONTHS)` if not present.
* `H1, W1` — 1 km grid size (expected 40×40 by default).
* `H30, W30` — 30 m grid size (expected 1333×1333 by default).
* `Y` — number of annual LC tiles (should equal `len(YEARS)`).
* `month_to_year` — `(T,)` torch.long tensor mapping month index → year index.
* Outputs saved per city:

  * `lc.pt` — `(T, 1, H30, W30)` monthly land-cover (replicated).
  * `month_to_year.pt` — `(T,)` index mapping.
  * `meta.json` — updated with `"files": {"lc": "lc.pt", ...}` and `"month_to_year_file"`.

---

## Important helper functions & behaviors

* `extract_zip_if_needed(zip_path, out_dir)`

  * If `zip_path` exists, extracts into the dirname of `out_dir`, then tries to heuristically find a folder containing multiple expected cities. Returns a candidate root or `out_dir`.
* `_load(name, expect_shape=None, make_float=False, make_mask=False)`

  * Loads tensor from path in `meta["files"]` or `{name}.pt`. Applies `_mask01` (nonzero→1 float) when `make_mask=True` or casts to float when `make_float=True`. Validates shape when `expect_shape` provided.
* `_mask01(t)` — converts any nonzero → 1.0 (float32), zeros remain 0.0.
* `_validate_shape(name, t, exp)` — intended to raise on mismatch. **Bug** (see below).

---

## Bug & one-line fix (important)

The helper `_validate_shape` currently does:

```python
if tuple(t.shape) != tuple(exp):
    raise ValueValue(f"{name} has shape {tuple(t.shape)} but expected {tuple(exp)}")
```

`ValueValue` is undefined and will raise a `NameError` if a shape mismatch occurs. Replace `ValueValue` with the standard `ValueError`. The one-line fix:

```python
# replace in _validate_shape
raise ValueError(f"{name} has shape {tuple(t.shape)} but expected {tuple(exp)}")
```

---

## Memory / storage note (practical)

Replicating annual 1333×1333 tiles to monthly creates a large tensor. approximate sizes:

* Per 1333×1333 tile = `1,333 × 1,333 = 1,776,889` pixels.
* If stored as `int16` (2 bytes/pixel): \~3.39 MiB per tile. For `T=216` → **\~732 MiB** total.
* If stored as `float32` (4 bytes/pixel): \~6.78 MiB per tile. For `T=216` → **\~1.46 GiB** total.

If disk or RAM is constrained, prefer keeping the 18 annual tiles and using `month_to_year` at runtime (no replication), or save `lc.pt` compressed or as `int16`.

---

## Design choices & caveats

* The script **explicitly duplicates** annual LC to align shapes with monthly variables — convenient but memory-heavy.
* The `month_to_year` array is defensive (`min(Y-1, t // 12)`) to avoid indexing out-of-range if `T` and `Y*12` mismatch.
* `extract_zip_if_needed` may place files in a sibling directory depending on ZIP structure — check its output if the script fails to find city folders.
* `_load` will error loudly if expected files are missing — this is helpful for debugging but can be softened to continue processing remaining cities if desired.
* `FORCE_FLOAT32` ensures numeric variables are float32; LC is intentionally *not* converted (keeps integer classes).

---

## Troubleshooting tips

* If you hit memory errors when creating `lc_monthly`:

  * Option 1: **Don't replicate** — read `lc_annual` on-the-fly using `month_to_year` during training/inference.
  * Option 2: Create and save `lc_monthly` in **chunks** (e.g., loop over smaller batches of months) instead of allocating the whole `(T, …)` tensor at once.
  * Option 3: Convert to `int16` (if not already) before repeat — it halves memory vs `float32`.
* If the script cannot find city folders after extraction, inspect the ZIP contents and/or set `ROOT_DIR` to the correct extracted folder.
* If you see unexpected shapes, double-check `meta.json` values for `grid_1km` / `grid_30m` and `t_steps`.
* If mask logic seems weak, consider using dataset-specific validity flags or NaN checks instead of simple `!= 0`.

