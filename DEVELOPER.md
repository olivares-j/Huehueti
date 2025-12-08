# Huehueti — Developer Notes

This short developer guide documents the expected input CSV format, required columns, units, missing-value handling, and quick commands to run Huehueti.

## Required CSV columns (header names)
The loader expects the following columns in the input CSV (exact names):

- Identifier
  - `source_id` (string or integer) — unique ID for each source (becomes index)
- Astrometry
  - `parallax` (float) — mas
  - `parallax_error` (float) — mas
- Photometry (values)
  - `g`, `bp`, `rp`, `gmag`, `rmag`, `imag`, `ymag`, `zmag`, `Jmag`, `Hmag`, `Kmag` (magnitudes)
- Photometry (errors)
  - `g_error`, `bp_error`, `rp_error`, `e_gmag`, `e_rmag`, `e_imag`, `e_ymag`, `e_zmag`, `e_Jmag`, `e_Hmag`, `e_Kmag` (magnitudes)

These names are defined in `Huehueti.observables`. If your catalog uses different names, either rename the columns or update `Huehueti.py`'s observables mapping accordingly.

## Units
- Photometry: magnitudes (`[mag]`)
- Parallax: milliarcseconds (`[mas]`)

## Missing data / error handling
- Zero error values in error columns are treated as missing and replaced by NaN before filling.
- By default missing error values are filled with the maximum value of the error column (use `fill_nan="mean"` in `load_data` to use the mean instead).
- If a photometric value is missing (NaN), its corresponding error is set back to NaN.

## Photometric limits filtering
- The loader filters out sources that are brighter than the isochrone model limits. The function `_isochrone_photometric_limits` reads `phot_min` from the serialized MLP (`file_mlp`) and converts it to apparent magnitudes using an average distance estimate (1000 / parallax mean).

## Files and directories
- Default directories used in the example `__main__`:
  - Data: `./data/` (e.g., `data/Pleiades.csv`)
  - MLPs: `./mlps/` (e.g., `mlps/PARSEC_10x96/mlp.pkl`)
  - Outputs: `./outputs/…/`
- Key outputs created by Huehueti (in `dir_out`):
  - `Identifiers.csv`, `Observations.nc`, `Chains.nc`, `Prior.nc`
  - Plots: `Posterior.pdf`, `Comparison_prior_posterior.pdf`, `Predictions.pdf`, `Color-magnitude_diagram.png`, etc.
  - Statistics: `Sources_statistics.csv`, `Global_statistics.csv`

## Quick run (example)
1. Place your input CSV at `data/YourCatalog.csv`.
2. Ensure you have an MLP dill file (e.g. `mlps/PARSEC_10x96/mlp.pkl`) with the `phot_min` and domain attributes consumed by the code.
3. Run:
   ```bash
   python Huehueti.py
   ```
   The example `__main__` in `Huehueti.py` uses `data/Pleiades.csv` and the MLP path `mlps/PARSEC_10x96/mlp.pkl`. Adjust these paths or run Huehueti programmatically from another script.

## Python dependencies
At minimum the project requires:
- Python 3.8+
- numpy
- pandas
- pymc
- arviz
- dill
- xarray
- seaborn
- matplotlib

## Notes and tips
- The MLP file must provide compatible domains (e.g., `age_domain`, `theta_domain`) and a `phot_min` array; the loader asserts that the age prior lies inside `mlp.age_domain`.
- If you modify the `observables` mapping in `Huehueti.py`, verify all downstream uses (plotting, summary) still reference the correct keys.

If you want, I can:
- Add a small script that validates a CSV against the required schema (checks presence of columns and some value sanity).
- Add type hints for class attributes (dataclass-like) or add a mypy config for the repository.