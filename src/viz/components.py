from __future__ import annotations

import pandas as pd
import streamlit as st


__all__ = [
    "CARD_STYLES",
    "TERRAIN_NOTE",
    "terrain_note",
    "VARIABLE_NOTES",
    "freezing_level_note",
    "swe_note",
    "CENTRED_ACC_NOTE",
    "ERA5_LINEAGE_NOTE",
    "COMPARISON_MAP_VIEWS",
    "PRECIP_COMPARISON_MAP_VIEWS",
    "PRECIP_NOTE",
    "render_comparison_map_toggle",
    "render_precip_cards",
    "render_threshold_selector",
    "inject_card_styles",
    "render_metric_cards",
    "render_lead_time_selector",
    "render_map_toggle",
    "render_map_view_toggle",
    "render_month_range_selector",
]

TERRAIN_NOTE = (
    "Mountain region: at 0.25° the forecast model and ERA5 represent terrain at different heights, "
    "so part of the temperature bias and error here reflects terrain representation, "
    "not forecast error."
)

_TERRAIN_NOTES = {
    "t2m": TERRAIN_NOTE,
    "msl": (
        "Mountain region: mean sea level pressure over high terrain is not measured but extrapolated "
        "down to sea level, and each model and ERA5 do the reduction differently, so part of the "
        "pressure bias and error here reflects that reduction, not forecast error."
    ),
    "ws10": (
        "Mountain region: at 0.25° the forecast model and ERA5 represent terrain height and surface "
        "roughness differently, so part of the 10 m wind speed bias and error here reflects terrain "
        "representation, not forecast error."
    ),
    "frzl": (
        "Mountain region: the freezing level is expressed above sea level while model and ERA5 terrain "
        "differ in height at 0.25°, and points where either source has the 0°C level at the surface are "
        "excluded, so part of the bias and error here reflects terrain representation, not forecast error."
    ),
    "swe": (
        "Mountain region: snow water equivalent depends on how each model and ERA5 represent terrain height "
        "and snow cover at 0.25°, and ERA5 snow is a weak truth in complex terrain and on glaciers, so part "
        "of the bias and error here reflects the reference and terrain representation, not forecast error."
    ),
    "tp24": (
        "Mountain region: precipitation in the forecast models and in ERA5 depends strongly on how each "
        "represents the terrain at 0.25°, and ERA5 precipitation is a weak truth in complex terrain, so "
        "part of the frequency bias, ETS and mean bias here reflects the reference and terrain "
        "representation, not forecast error."
    ),
}


def terrain_note(variable: str) -> str | None:
    """Caveat for a mountain region, or None when the variable has none (500 hPa height)."""
    return _TERRAIN_NOTES.get(variable)


# How each variable is verified, shown under "How these numbers are computed".
VARIABLE_NOTES = {
    "msl": "Mean sea level pressure is the models' own reduction to sea level, compared with ERA5's.",
    "z500": "500 hPa geopotential height (metres): the headline measure of large-scale forecast skill.",
    "ws10": (
        "10 m wind speed is computed from the u and v components in each forecast and in ERA5 *before* "
        "verification, so bias and RMSE are for the scalar speed, not the vector. The ACC climatology is "
        "the mean of ERA5 wind *speed* (not the speed of the mean wind), 1991–2020."
    ),
    "swe": (
        "Snow water equivalent (mm of water) is the GFS water equivalent of the snow on the ground (`WEASD`) "
        "compared with ERA5 snow depth (given in metres of water equivalent). GFS has no value over open "
        "water, so those points are left out of the sample. ERA5 cells whose snow never melts out (ERA5 "
        "glaciers, holding 1–10 m of water all year; 139 cells in western North America, none in Europe) "
        "are also left out: an ERA5 Aug–Sep minimum of at least 10 mm in every year 2015–2026. The "
        "excluded share is shown. ECMWF is not "
        "included: its open data has snow depth only from November 2025. ERA5 snow is a model product and a "
        "weaker truth than ERA5 temperature or pressure, especially in complex terrain and on glaciers."
    ),
    "tp24": (
        "24h precipitation is the total over the 24 hours *ending* at the lead time, from each model's "
        "accumulation (the accumulation at the lead minus the one 24 h earlier) and, in ERA5, the sum of "
        "the 24 hourly values. An event is a total at or above the threshold in the forecast and in ERA5. "
        "Frequency bias is forecast events ÷ observed events (1 = unbiased); ETS is the equitable threat "
        "score, hits corrected for those expected by chance (1 = perfect, 0 = no skill beyond chance). "
        "Counts are weighted by cos(latitude) and pooled over grid points and forecasts. ERA5 precipitation "
        "is a model product, a weaker truth than ERA5 temperature or pressure, especially in complex terrain."
    ),
    "frzl": (
        "Freezing level is the height of the 0°C level above sea level: the GFS highest tropospheric "
        "freezing level, compared with ERA5's zero-degree level (given above ground, converted with ERA5 "
        "orography). A point and forecast are verified only if both put the freezing level above their own "
        "surface; sub-zero surfaces are excluded from both. Because that sample varies by point and time, "
        "only bias and RMSE are shown (no ACC). The GFS half of the mask depends on the forecast, so the "
        "excluded share grows with lead time and the curves are not a strictly homogeneous sample."
    ),
}


def swe_note(excluded_fraction: float) -> str:
    """Sample note for snow water equivalent, with the excluded share of this region and window."""
    return (f"Snow water equivalent: {excluded_fraction:.1%} of this region's grid-point forecasts are "
            "excluded because GFS has no snow value there (open water) or ERA5 has permanent snow (glaciers).")


def freezing_level_note(masked_fraction: float) -> str:
    """Sample note for the freezing level, with the excluded share of this region and window."""
    return (f"Freezing level: {masked_fraction:.1%} of this region's grid-point forecasts are excluded "
            "because ERA5 or GFS (or both) has the 0°C level at the surface (sub-zero surface).")


PRECIP_NOTE = (
    "Precipitation events are 24h totals at or above the selected threshold, in the forecast and in ERA5. "
    "Frequency bias is forecast events ÷ observed events (1 = unbiased); ETS is 1 for a perfect forecast "
    "and 0 for no skill beyond chance. There is no ACC or RMSE for precipitation."
)

CENTRED_ACC_NOTE = (
    "ACC is centred: each forecast's regional mean anomaly is removed before correlating. "
    "Smaller regions have less spatial anomaly variance, so their ACC is lower and reaches the "
    "0.6 line at shorter lead times than larger regions."
)

ERA5_LINEAGE_NOTE = (
    "ERA5 is produced with the ECMWF model. Verifying ECMWF against ERA5 favours ECMWF, "
    "so this comparison is not model-neutral."
)

# Single CSS block — injected once by inject_card_styles(), not per card.
CARD_STYLES = """
<style>
.metric-card {
    border-radius: 8px;
    padding: 14px 12px;
    text-align: center;
    font-family: sans-serif;
    border: 1px solid var(--card-border, #dde1ea);
    background: var(--card-bg, #f0f2f6);
    color: var(--card-fg, #262730);
    margin-bottom: 4px;
}
.metric-card .mc-label {
    font-size: 0.70rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    opacity: 0.70;
    margin-bottom: 5px;
}
.metric-card .mc-value {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-card .mc-sub {
    font-size: 0.78rem;
    margin-top: 4px;
    opacity: 0.80;
}
</style>
"""

_CARD_HTML = (
    '<div class="metric-card" style="'
    "background:{bg};color:{fg};border-color:{border};"
    '">'
    '<div class="mc-label">{label}</div>'
    '<div class="mc-value">{value}</div>'
    '<div class="mc-sub">{sub}</div>'
    "</div>"
)


def inject_card_styles() -> None:
    """Inject metric-card CSS once into the page. Call from app.py at startup."""
    st.markdown(CARD_STYLES, unsafe_allow_html=True)


def _card(
    label: str,
    value: str,
    sub: str = "",
    bg: str = "#f0f2f6",
    fg: str = "#262730",
    border: str = "#dde1ea",
) -> str:
    return _CARD_HTML.format(
        label=label, value=value, sub=sub, bg=bg, fg=fg, border=border
    )


# Bias below this size (in the variable's units) is shown as "near zero".
_NEAR_ZERO_BIAS = {"t2m": 0.05, "msl": 0.05, "z500": 0.5, "ws10": 0.05, "frzl": 5.0, "tp24": 0.02, "swe": 0.5}
_BIAS_WORDS = {"t2m": ("Warm bias", "Cold bias")}


def render_metric_cards(summary_df: pd.DataFrame, lead_time_h: int, units: str = "°C",
                        title: str | None = None, variable: str = "t2m", has_acc: bool = True) -> None:
    """Render metric summary cards for the selected lead time.

    summary_df: columns lead_time_h, acc_c, rmse, bias (metrics.region_metrics).
    title: heading above the cards (comparison mode shows one row per model).
    has_acc: False for variables without an ACC score (freezing level): ACC and the
    skill-threshold card are left out.
    """
    if title:
        st.markdown(f"**{title}**")
    candidates = summary_df[summary_df["lead_time_h"] == lead_time_h]
    if candidates.empty:
        idx = (summary_df["lead_time_h"] - lead_time_h).abs().idxmin()
        row = summary_df.loc[idx]
    else:
        row = candidates.iloc[0]

    acc = float(row["acc_c"])
    rmse = float(row["rmse"])
    bias = float(row["bias"])
    skilled = acc >= 0.6

    cols = st.columns(4 if has_acc else 2)
    c_acc, c_rmse, c_bias, c_skill = cols if has_acc else (None, *cols, None)

    if c_acc is not None:
        with c_acc:
            st.markdown(
                _card("ACC (centred)", f"{acc:.3f}", sub=f"at {lead_time_h}h lead"),
                unsafe_allow_html=True,
            )
    with c_rmse:
        st.markdown(
            _card("RMSE", f"{rmse:.2f} {units}", sub=f"at {lead_time_h}h lead"),
            unsafe_allow_html=True,
        )
    with c_bias:
        high, low = _BIAS_WORDS.get(variable, ("High bias", "Low bias"))
        near = _NEAR_ZERO_BIAS.get(variable, 0.05)
        bias_desc = high if bias > near else low if bias < -near else "Near-zero bias"
        st.markdown(
            _card("Bias", f"{bias:+.2f} {units}", sub=bias_desc),
            unsafe_allow_html=True,
        )
    if c_skill is not None:
        with c_skill:
            if skilled:
                card = _card("Skill Threshold", "Above", sub=f"ACC {acc:.3f} ≥ 0.6",
                             bg="#d4edda", fg="#155724", border="#c3e6cb")
            else:
                card = _card("Skill Threshold", "Below", sub=f"ACC {acc:.3f} < 0.6",
                             bg="#f8d7da", fg="#721c24", border="#f5c6cb")
            st.markdown(card, unsafe_allow_html=True)


def render_precip_cards(summary_df: pd.DataFrame, lead_time_h: int, threshold: float, units: str = "mm",
                        title: str | None = None) -> None:
    """Cards for precipitation at one lead time and threshold: frequency bias, ETS, mean bias, and the
    observed event frequency (how often ERA5 reaches the threshold, for context).

    summary_df: columns lead_time_h, freq_bias, ets, bias, obs_frequency (one threshold).
    """
    if title:
        st.markdown(f"**{title}**")
    candidates = summary_df[summary_df["lead_time_h"] == lead_time_h]
    row = (candidates.iloc[0] if not candidates.empty
           else summary_df.loc[(summary_df["lead_time_h"] - lead_time_h).abs().idxmin()])
    fb, ets_, bias, obs_freq = (float(row[c]) for c in ("freq_bias", "ets", "bias", "obs_frequency"))
    sub = f"≥ {threshold:g} {units}/24h at {lead_time_h}h lead"

    def fmt(x: float, f: str) -> str:
        return "n/a" if x != x else format(x, f)

    c_fb, c_ets, c_bias, c_obs = st.columns(4)
    with c_fb:
        word = "n/a" if fb != fb else "Over-forecast" if fb > 1.05 else "Under-forecast" if fb < 0.95 else "Near unbiased"
        st.markdown(_card("Frequency bias", fmt(fb, ".2f"), sub=f"{word} events"), unsafe_allow_html=True)
    with c_ets:
        st.markdown(_card("ETS", fmt(ets_, ".3f"), sub=sub), unsafe_allow_html=True)
    with c_bias:
        near = _NEAR_ZERO_BIAS["tp24"]
        word = "Too wet" if bias > near else "Too dry" if bias < -near else "Near-zero bias"
        st.markdown(_card("Mean bias", f"{bias:+.2f} {units}/24h", sub=word), unsafe_allow_html=True)
    with c_obs:
        st.markdown(_card("ERA5 event frequency", fmt(100 * obs_freq, ".1f") + " %", sub=sub),
                    unsafe_allow_html=True)


def render_threshold_selector(thresholds: list[float], units: str = "mm", key: str = "threshold") -> float:
    """Precipitation threshold selector (config ``thresholds``). Returns the chosen threshold."""
    labels = {f"{t:g} {units}/24h": t for t in thresholds}
    return labels[st.radio("Precipitation threshold", list(labels), key=key)]


def render_lead_time_selector(lead_times: list[int], key: str = "lt") -> int:
    """Segmented button row for lead time selection. Returns selected lead time in hours."""
    options = [f"{lt}h" for lt in lead_times]
    selected = st.segmented_control(
        "Lead Time",
        options=options,
        default=options[0],
        key=key,
    )
    if selected is None:
        return lead_times[0]
    return int(selected.replace("h", ""))


def render_map_toggle(key: str = "map_metric", has_acc: bool = True, categorical: bool = False) -> str:
    """Toggle for spatial map metric. Returns 'rmse', 'bias', or 'acc' (not offered without ACC); for
    precipitation (``categorical``) 'freq_bias', 'ets' or 'bias' (mean bias)."""
    labels = ({"Frequency bias": "freq_bias", "ETS": "ets", "Mean bias": "bias"} if categorical else
              {"RMSE": "rmse", "Bias": "bias"} | ({"ACC Skill": "acc"} if has_acc else {}))
    selected = st.radio(
        "Map view",
        list(labels.keys()),
        horizontal=False,
        key=key,
    )
    return labels[selected]


# Comparison-mode map views: key -> label. {a} and {b} are the two model names.
COMPARISON_MAP_VIEWS = {
    "rmse_diff": "RMSE difference ({a} − {b})",
    "acc_diff": "ACC difference ({b} − {a})",
    "bias:a": "Bias — {a}",
    "bias:b": "Bias — {b}",
}


# The same for precipitation: ETS difference oriented like ACC (B − A: positive = B better).
PRECIP_COMPARISON_MAP_VIEWS = {
    "ets_diff": "ETS difference ({b} − {a})",
    "freq_bias:a": "Frequency bias — {a}",
    "freq_bias:b": "Frequency bias — {b}",
    "bias:a": "Mean bias — {a}",
    "bias:b": "Mean bias — {b}",
}


def render_comparison_map_toggle(model_a: str, model_b: str, key: str = "cmp_map_metric",
                                 views: dict[str, str] | None = None) -> str:
    """Map view for a head-to-head. Returns 'rmse_diff', 'acc_diff', 'bias:a' or 'bias:b' (``views`` =
    PRECIP_COMPARISON_MAP_VIEWS: 'ets_diff', 'freq_bias:a' …)."""
    views = views or COMPARISON_MAP_VIEWS
    labels = {v.format(a=model_a.upper(), b=model_b.upper()): k for k, v in views.items()}
    return labels[st.radio("Map view", list(labels), key=key)]


def render_map_view_toggle(key: str = "map_full_domain") -> bool:
    """True when the map should show the whole domain instead of zooming to the region."""
    return st.toggle("Show full domain on map", value=False, key=key,
                     help="Off: the map zooms to the selected region. On: whole domain with the region boxed.")


def render_month_range_selector(months: list[str], key: str = "month_range") -> tuple[str, str]:
    """Month-resolution analysis period. ``months`` are 'YYYY-MM' strings in order.

    Returns (first_month, last_month), inclusive. With a single month available
    there is nothing to choose, so it is shown as text.
    """
    st.markdown("**Analysis period**")
    if len(months) == 1:
        st.caption(months[0])
        return months[0], months[0]
    first, last = st.select_slider(
        "Select months",
        options=months,
        value=(months[0], months[-1]),
        key=key,
        label_visibility="collapsed",
    )
    return first, last
