from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils.config import Bounds

__all__ = [
    "CARD_STYLES",
    "inject_card_styles",
    "render_metric_cards",
    "render_lead_time_selector",
    "render_region_selector",
    "render_map_toggle",
    "render_map_view_toggle",
    "render_month_range_selector",
]

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


def render_metric_cards(summary_df: pd.DataFrame, lead_time_h: int, units: str = "°C") -> None:
    """Render four metric summary cards for the selected lead time.

    summary_df: columns lead_time_h, acc_c, rmse, bias (metrics.region_metrics).
    """
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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            _card("ACC (centred)", f"{acc:.3f}", sub=f"at {lead_time_h}h lead"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _card("RMSE", f"{rmse:.2f} {units}", sub=f"at {lead_time_h}h lead"),
            unsafe_allow_html=True,
        )
    with c3:
        bias_desc = (
            "Warm bias" if bias > 0.05
            else "Cold bias" if bias < -0.05
            else "Near-zero bias"
        )
        st.markdown(
            _card("Bias", f"{bias:+.2f} {units}", sub=bias_desc),
            unsafe_allow_html=True,
        )
    with c4:
        if skilled:
            st.markdown(
                _card(
                    "Skill Threshold", "Above",
                    sub=f"ACC {acc:.3f} ≥ 0.6",
                    bg="#d4edda", fg="#155724", border="#c3e6cb",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _card(
                    "Skill Threshold", "Below",
                    sub=f"ACC {acc:.3f} < 0.6",
                    bg="#f8d7da", fg="#721c24", border="#f5c6cb",
                ),
                unsafe_allow_html=True,
            )


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


def render_region_selector(
    regions: dict[str, Bounds], key: str = "region"
) -> tuple[str, dict[str, float]]:
    """Dropdown region selector over config regions. Returns (region_name, bounds_dict)."""
    region_name = st.selectbox("Region", list(regions), key=key)
    b = regions[region_name]
    return region_name, {
        "lat_min": b.lat_min, "lat_max": b.lat_max,
        "lon_min": b.lon_min, "lon_max": b.lon_max,
    }


def render_map_toggle(key: str = "map_metric") -> str:
    """Toggle for spatial map metric. Returns 'rmse', 'bias', or 'acc'."""
    labels = {"RMSE": "rmse", "Bias": "bias", "ACC Skill": "acc"}
    selected = st.radio(
        "Map view",
        list(labels.keys()),
        horizontal=False,
        key=key,
    )
    return labels[selected]


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
