# Rice Rhythm — water rhythm of rice fields from NISAR L-band radar

NASA Space Apps Challenge 2026 · challenge **"Dancing with the SARs"**

Rice Rhythm reads a season (June–October 2026) of **NISAR L2 GCOV** radar images (L-band, HH and HV)
and builds the "water rhythm" of every rice field: was the field kept flooded all season, was the water
drained and re-flooded (alternate wetting and drying, AWD), or was rice sown without early flooding
(direct-seeded rice, DSR). L-band sees water **under the rice canopy**, where optical satellites only see leaves.

> NISAR data courtesy of NASA/JPL, ASF DAAC.

## How it works

1. `fetch_nisar.py` searches ASF for NISAR GCOV over the area (`asf_search`), keeps one track and
   reads **only the needed window** of the HH/HV layers over HTTPS (h5py + fsspec with an Earthdata
   Bearer token) — no 2 GB downloads. Saves `data*/stack.nc`.
2. `analyze_fields.py` — per field (from `fields.geojson`, or 300 m grid cells) builds HH/HV time series and
   classifies with simple, data-tuned rules:
   - **rice**: HV (vegetation) grows ≥ 3 dB after mid-July;
   - **early water**: HH below −16.5 dB before 15 July (open water is dark for radar);
   - **drainage event**: after mid-July, HH drops by more than the noise level (99th percentile of
     date-to-date change on non-rice cells, after removing the scene-wide shift) and rises back;
   - **DSR** (optional `--dsr`): rice without early water while neighbours are flooded.
   Outputs charts, `compare.png`, an interactive `map.html` (folium), `summary.txt`, `timeseries.csv`
   and sonification of each field rhythm (`--sound`).
3. `thresholds.py` — distributions used to choose thresholds; `montage.py` — 5 random charts per class for manual check.
4. `validate_s2.py` — cross-check with **Sentinel-2 L2A** (Microsoft Planetary Computer): MNDWI water
   vs radar water on cloud-free dates within ±3 days.
5. `make_visuals.py` (per-date images, `season.gif`, `before_after.png`), `make_slides.py` (1920×1080 slides).

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EARTHDATA_TOKEN=...            # https://urs.earthdata.nasa.gov → Generate Token (never commit it)

python fetch_nisar.py --region kzyl --list-only
python fetch_nisar.py --region kzyl                     # → data/stack.nc
python make_visuals.py --stack data/stack.nc --out out --region "Zhalagash"
python analyze_fields.py --stack data/stack.nc --out out --sound
python validate_s2.py --stack data/stack.nc --out out
python make_slides.py --stack data/stack.nc --out out --region "Zhalagash" --prefix kzyl
```

Other area: `python fetch_nisar.py --region karnal --tracks` lists tracks with dates; then
`--region karnal --track N --direction A|D`. Areas are defined in `REGIONS` in `fetch_nisar.py`.
A step-by-step guide in Russian is in `README_run_ru.md`.

## Data sources

- **NISAR L2 GCOV** (NASA-ISRO SAR mission), via **ASF DAAC** — *NISAR data courtesy of NASA/JPL, ASF DAAC.*
- **Sentinel-2 L2A** (Copernicus/ESA), via **Microsoft Planetary Computer** — validation only.
- Basemap in `map.html`: Esri World Imagery / OpenStreetMap.

## Limitations

- Without field outlines the area is split into 300 m cells, which can mix paddies with dikes and roads.
- 12-day revisit: short drainage events can fall between acquisitions.
- Radar "water" (low HH) is only valid early in the season: after harvest smooth dry soil is also dark.

## Acknowledgements
Data: NISAR L2 GCOV (NASA/JPL, ASF DAAC), Sentinel-2 L2A (ESA, via Microsoft Planetary Computer).
An AI coding assistant was used to help write code and edit text. The team chose the problem, region and data, made the analysis decisions and validated the results.
