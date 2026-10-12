"""Regenerate the README figures in docs/figures/ from the committed dashboard data.

    python scripts/make_readme_figures.py

Both figures are built with the dashboard's own code (src.viz.dashboard_data and the src.viz figure
builders), in its GFS-vs-ECMWF comparison mode:
  skill_vs_lead_t2m_western_europe.png   skill curves, all paired months
  rmse_diff_t2m_europe_24h.png           RMSE GFS − ECMWF map, 24 h lead, European Alps box highlighted

Static export needs kaleido (requirements-dev.txt) and a Chrome/Chromium on this machine: kaleido 1.x
drives an installed browser and does not bundle one. Map files come from data/dashboard/maps when present,
else from the release download (src.data.map_store).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import comparison_models, load_config
from src.viz.dashboard_data import (
    difference_map,
    load_coverage,
    load_map_stats,
    load_region_table,
    paired_run_count,
    region_summary,
)
from src.viz.skill_curves import build_comparison_curves
from src.viz.spatial_maps import build_spatial_map

OUT_DIR = REPO_ROOT / "docs" / "figures"
COMPARISON = "gfs_vs_ecmwf"
VARIABLE = "t2m"
DOMAIN = "europe"
CURVE_REGION = "Western Europe"
MAP_REGION = "European Alps"
MAP_LEAD_H = 24
MAP_HEIGHT = 760
WIDTH, SCALE = 1100, 2          # px wide, pixel ratio


def prepare_curves(table, coverage, cfg):
    """Summaries per model (as the dashboard's comparison curves), paired run count and month span."""
    pair = comparison_models(COMPARISON)
    source = table[table.comparison == COMPARISON]
    paired_cov = coverage[coverage.comparison == COMPARISON]
    sel = (source.model == pair[0]) & (source.variable == VARIABLE) & (source.domain == DOMAIN)
    months = sorted(source.loc[sel & (source.region == CURVE_REGION), "month"].unique())
    summaries = {m: region_summary(source, m, VARIABLE, DOMAIN, CURVE_REGION, months) for m in pair}
    runs = paired_run_count(paired_cov, pair[0], COMPARISON, VARIABLE, DOMAIN, months)
    return summaries, runs, months


def prepare_map(table, cfg):
    """RMSE difference DataArray (GFS − ECMWF), title, colour-bar note and the map years used."""
    pair = comparison_models(COMPARISON)
    source = table[table.comparison == COMPARISON]
    months = source.loc[(source.model == pair[0]) & (source.variable == VARIABLE) & (source.domain == DOMAIN),
                        "month"].unique()
    years = tuple(sorted({int(m[:4]) for m in months}))
    stats = {m: load_map_stats(m, VARIABLE, DOMAIN, years, COMPARISON) for m in pair}
    if any(v is None for v in stats.values()):
        raise SystemExit("map statistics unavailable")
    da, title, note = difference_map(stats[pair[0]], stats[pair[1]], pair[0].upper(), pair[1].upper(), "rmse_diff")
    return da, title, note, years


def _label(pair) -> str:
    return " vs ".join(m.upper() for m in pair)


def main() -> None:
    cfg = load_config()
    table, coverage = load_region_table(), load_coverage()
    pair = comparison_models(COMPARISON)
    units = cfg.variables[VARIABLE].units
    long_name = cfg.variables[VARIABLE].long_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1
    summaries, runs, months = prepare_curves(table, coverage, cfg)
    print(f"Figure 1 — {len(months)} months {months[0]} to {months[-1]}, {runs:,} paired runs")
    print("lead_h  model   ACC(c)   RMSE    bias")
    for m, df in summaries.items():
        for r in df.itertuples():
            print(f"{r.lead_time_h:>6}  {m.upper():<6}  {r.acc_c:.3f}  {r.rmse:.3f}  {r.bias:+.3f}")
    title = (f"{_label(pair)} — {long_name} — {CURVE_REGION}, {months[0]} to {months[-1]}, "
             f"{runs:,} paired runs")
    fig = build_comparison_curves(summaries, CURVE_REGION, lead_times=cfg.lead_times_hours, units=units,
                                  title=title)
    fig.update_layout(title_font_size=14, margin=dict(l=70, r=40, t=130, b=60),
                      legend=dict(y=1.13))
    path1 = OUT_DIR / "skill_vs_lead_t2m_western_europe.png"
    fig.write_image(path1, width=WIDTH, height=fig.layout.height, scale=SCALE)

    # Figure 2
    da, name, note, years = prepare_map(table, cfg)
    d = cfg.domains[DOMAIN].bounds
    view = {"lat_min": d.lat_min, "lat_max": d.lat_max, "lon_min": d.lon_min, "lon_max": d.lon_max}
    b = cfg.regions(DOMAIN)[MAP_REGION]
    region_bounds = {"lat_min": b.lat_min, "lat_max": b.lat_max, "lon_min": b.lon_min, "lon_max": b.lon_max}
    fig = build_spatial_map(da, "rmse_diff", MAP_LEAD_H, region_bounds, MAP_REGION, units, view,
                            title=name, colorbar_note=note, variable=VARIABLE)
    field = da.sel(lead_time=MAP_LEAD_H, method="nearest")
    drawn = len(fig.data[0].lat)
    print(f"Figure 2 — {name}, {MAP_LEAD_H} h, map years {', '.join(map(str, years))}: "
          f"min {float(field.min()):+.3f}, max {float(field.max()):+.3f} {units}, {drawn:,} cells drawn")
    fig.update_layout(
        title_text=f"{name} — {long_name} — {MAP_LEAD_H}h lead, {DOMAIN.title()}, "
                   f"{', '.join(map(str, years))} ({MAP_REGION} outlined)")
    path2 = OUT_DIR / "rmse_diff_t2m_europe_24h.png"
    fig.write_image(path2, width=WIDTH, height=MAP_HEIGHT, scale=SCALE)

    for p in (path1, path2):
        print(f"wrote {p.relative_to(REPO_ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
