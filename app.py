"""
Paris ATM / FX Opportunity Explorer
Interactive Streamlit app for analyzing ATM placement,
FX advertising, and FX bureau opportunities across Paris,
Vincennes, and Boulogne-Billancourt.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import geopandas as gpd

st.set_page_config(page_title="Paris ATM / FX Opportunity Explorer", layout="wide")

# ─── Map presets ─────────────────────────────────────────────────────
MAP_PRESETS = {
    "All study area":    {"center": [48.855, 2.345], "zoom": 12},
    "Central Paris":     {"center": [48.862, 2.340], "zoom": 13},
    "Vincennes focus":   {"center": [48.847, 2.435], "zoom": 14},
    "Boulogne focus":    {"center": [48.842, 2.240], "zoom": 14},
    "Montmartre focus":  {"center": [48.885, 2.340], "zoom": 14},
    "Marais / Temple":   {"center": [48.860, 2.360], "zoom": 14},
}

CANDIDATE_COLORS = {
    "ATM":       "#e74c3c",
    "FX Ad":     "#2980b9",
    "FX Bureau": "#8e44ad",
}


# ─── Data loading (cached) ──────────────────────────────────────────
@st.cache_data
def load_candidates():
    atm = pd.read_csv("atm_candidates.csv")
    atm["type"] = "ATM"
    atm.rename(columns={"atm_score": "score"}, inplace=True)

    fx_ad = pd.read_csv("fx_ad_candidates.csv")
    fx_ad["type"] = "FX Ad"
    fx_ad.rename(columns={"fx_ad_score": "score"}, inplace=True)

    fx_bur = pd.read_csv("fx_bureau_candidates.csv")
    fx_bur["type"] = "FX Bureau"
    fx_bur.rename(columns={"fx_bureau_score": "score"}, inplace=True)

    return atm, fx_ad, fx_bur


@st.cache_data
def load_existing_atms():
    gdf = gpd.read_file("atm_paris.geojson").to_crs("EPSG:4326")
    return gdf


@st.cache_data
def load_existing_fx():
    gdf = gpd.read_file("fx_bureaux.geojson").to_crs("EPSG:4326")
    return gdf


@st.cache_data
def load_hotels():
    gdf = gpd.read_file("hotels.geojson").to_crs("EPSG:4326")
    return gdf


@st.cache_data
def load_attractions():
    with open("touristical_attraction.geojson", "r", encoding="utf-8") as f:
        data = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    return gdf


@st.cache_data
def load_datatourisme():
    dt = pd.read_csv("datatourisme.csv", encoding="utf-8")
    gdf = gpd.GeoDataFrame(
        dt, geometry=gpd.points_from_xy(dt.Longitude, dt.Latitude), crs="EPSG:4326"
    )
    return gdf


@st.cache_data
def load_zti():
    gdf = gpd.read_file("zones-touristiques-internationales.geojson").to_crs("EPSG:4326")
    return gdf


@st.cache_data
def load_airbnb_heatdata():
    """Load Airbnb data as lightweight heatmap points."""
    df = pd.read_csv("processed_fx_target_data.csv",
                      usecols=["latitude", "longitude", "foreign_tourist_ratio_pct",
                               "number_of_reviews", "occupancy_rate_pct"])
    # Weight by foreign ratio * review activity
    df["weight"] = (
        df["foreign_tourist_ratio_pct"].fillna(0) / 100 *
        np.log1p(df["number_of_reviews"]) *
        df["occupancy_rate_pct"].fillna(50) / 100
    )
    # Subsample for performance if very large
    if len(df) > 20000:
        df = df.sample(20000, random_state=42, weights=df["weight"].clip(0.01))
    return df[["latitude", "longitude", "weight"]].values.tolist()


# ─── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("ATM / FX Opportunity Explorer")
st.sidebar.markdown("**Paris - Vincennes - Boulogne**")

st.sidebar.markdown("---")
st.sidebar.subheader("Geographic Focus")
geo_focus = st.sidebar.selectbox("Focus area", list(MAP_PRESETS.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Candidate Filters")

candidate_types = st.sidebar.multiselect(
    "Candidate types to show",
    ["ATM", "FX Ad", "FX Bureau"],
    default=["ATM", "FX Ad"]
)

min_score = st.sidebar.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
top_n = st.sidebar.slider("Top N candidates per type", 1, 15, 15)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Layers")

show_existing_atms = st.sidebar.checkbox("Existing ATMs", value=False)
show_existing_fx = st.sidebar.checkbox("Existing FX bureaux", value=False)
show_hotels = st.sidebar.checkbox("Hotels", value=False)
show_attractions = st.sidebar.checkbox("Tourist attractions", value=False)
show_datatourisme = st.sidebar.checkbox("DATAtourisme points", value=False)
show_zti = st.sidebar.checkbox("Tourism zones (ZTI)", value=True)
show_heatmap = st.sidebar.checkbox("Demand heatmap (Airbnb/foreign)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
cluster_markers = st.sidebar.checkbox("Cluster existing supply markers", value=True)


# ─── Build map ──────────────────────────────────────────────────────
preset = MAP_PRESETS[geo_focus]
m = folium.Map(
    location=preset["center"],
    zoom_start=preset["zoom"],
    tiles="CartoDB positron",
    control_scale=True,
)

# Widen the map bounds to always include Vincennes + Boulogne
m.fit_bounds([[48.80, 2.18], [48.91, 2.48]])
# But if a focus is selected, use its center/zoom instead
if geo_focus != "All study area":
    m.location = preset["center"]
    m.options["zoom"] = preset["zoom"]
    # Remove fit_bounds for focused views
    m._children = {k: v for k, v in m._children.items() if "fit_bounds" not in k}

# ─── ZTI zones ──────────────────────────────────────────────────────
if show_zti:
    zti = load_zti()
    for _, row in zti.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x: {
                "fillColor": "#f39c12",
                "color": "#e67e22",
                "weight": 2,
                "fillOpacity": 0.15,
            },
            tooltip=row.get("name", "ZTI Zone"),
        ).add_to(m)

# ─── Demand heatmap ─────────────────────────────────────────────────
if show_heatmap:
    heat_data = load_airbnb_heatdata()
    HeatMap(
        heat_data,
        radius=12,
        blur=18,
        max_zoom=14,
        gradient={0.2: "blue", 0.4: "cyan", 0.6: "lime", 0.8: "yellow", 1.0: "red"},
    ).add_to(m)

# ─── Existing ATMs ──────────────────────────────────────────────────
if show_existing_atms:
    atms_gdf = load_existing_atms()
    if cluster_markers:
        atm_cluster = MarkerCluster(name="Existing ATMs")
        for _, row in atms_gdf.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=4,
                color="#7f8c8d",
                fill=True,
                fill_opacity=0.6,
                tooltip=f"ATM: {row.get('name', 'N/A')}",
            ).add_to(atm_cluster)
        atm_cluster.add_to(m)
    else:
        for _, row in atms_gdf.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=4,
                color="#7f8c8d",
                fill=True,
                fill_opacity=0.6,
                tooltip=f"ATM: {row.get('name', 'N/A')}",
            ).add_to(m)

# ─── Existing FX bureaux ────────────────────────────────────────────
if show_existing_fx:
    fx_gdf = load_existing_fx()
    if cluster_markers:
        fx_cluster = MarkerCluster(name="Existing FX Bureaux")
        for _, row in fx_gdf.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=5,
                color="#9b59b6",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"FX: {row.get('name', 'N/A')}",
            ).add_to(fx_cluster)
        fx_cluster.add_to(m)
    else:
        for _, row in fx_gdf.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=5,
                color="#9b59b6",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"FX: {row.get('name', 'N/A')}",
            ).add_to(m)

# ─── Hotels ─────────────────────────────────────────────────────────
if show_hotels:
    hotels = load_hotels()
    hotel_cluster = MarkerCluster(name="Hotels")
    for _, row in hotels.iterrows():
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=3,
            color="#3498db",
            fill=True,
            fill_opacity=0.4,
            tooltip=f"Hotel: {row.get('name', 'N/A')}",
        ).add_to(hotel_cluster)
    hotel_cluster.add_to(m)

# ─── Tourist attractions ────────────────────────────────────────────
if show_attractions:
    attrs = load_attractions()
    for _, row in attrs.iterrows():
        name = row.get("name", row.get("name:en", "Attraction"))
        folium.Marker(
            [row.geometry.y, row.geometry.x],
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
            tooltip=name,
        ).add_to(m)

# ─── DATAtourisme ───────────────────────────────────────────────────
if show_datatourisme:
    dt = load_datatourisme()
    dt_cluster = MarkerCluster(name="DATAtourisme")
    for _, row in dt.iterrows():
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=4,
            color="#1abc9c",
            fill=True,
            fill_opacity=0.5,
            tooltip=f"{row.get('Nom', 'N/A')} ({row.get('Sous-type', '')})",
        ).add_to(dt_cluster)
    dt_cluster.add_to(m)

# ─── Candidate opportunities ────────────────────────────────────────
atm_cands, fx_cands, fx_bur_cands = load_candidates()
all_cands = pd.concat([atm_cands, fx_cands, fx_bur_cands], ignore_index=True)

# Filter
filtered = all_cands[
    (all_cands["type"].isin(candidate_types)) &
    (all_cands["score"] >= min_score) &
    (all_cands["rank"] <= top_n)
].copy()

for _, row in filtered.iterrows():
    ctype = row["type"]
    color = CANDIDATE_COLORS.get(ctype, "#333333")
    icon_map = {"ATM": "money", "FX Ad": "bullhorn", "FX Bureau": "exchange"}

    popup_html = f"""
    <div style="font-family:Arial;font-size:13px;min-width:250px">
        <h4 style="margin:0 0 6px;color:{color}">{ctype} Candidate #{int(row['rank'])}</h4>
        <b>Area:</b> {row.get('neighbourhood', 'N/A')}<br>
        <b>Score:</b> {row['score']:.3f}<br>
        <hr style="margin:4px 0">
        <b>Score components:</b><br>
        &bull; Foreign demand: {row.get('n_foreign', 0):.2f}<br>
        &bull; Airbnb density: {row.get('n_airbnb', 0):.2f}<br>
        &bull; Hotel signal: {row.get('n_hotels', 0):.2f}<br>
        &bull; Attractions: {row.get('n_attractions', 0):.2f}<br>
        &bull; ZTI proximity: {row.get('n_zti', 0):.2f}<br>
        &bull; Commercial: {row.get('n_commercial', 0):.2f}<br>
        &bull; ATM scarcity: {row.get('n_atm_scarcity', 0):.2f}<br>
        &bull; FX scarcity: {row.get('n_fx_scarcity', 0):.2f}<br>
        <hr style="margin:4px 0">
        <b>Nearby supply:</b><br>
        &bull; ATMs in 500m: {int(row.get('atm_count_500m', 0))}<br>
        &bull; FX bureaux in 500m: {int(row.get('fx_bureau_count_500m', 0))}<br>
        <b>Nearby demand:</b><br>
        &bull; Airbnb listings: {int(row.get('airbnb_count', 0))}<br>
        &bull; Hotels: {int(row.get('hotel_count', 0))}<br>
        <hr style="margin:4px 0">
        <b>Coords:</b> {row['center_lat']:.5f}, {row['center_lon']:.5f}<br>
        <i>{row.get('business_explanation', '')}</i>
    </div>
    """

    folium.Marker(
        [row["center_lat"], row["center_lon"]],
        popup=folium.Popup(popup_html, max_width=320),
        icon=folium.Icon(
            color="red" if ctype == "ATM" else ("blue" if ctype == "FX Ad" else "purple"),
            icon=icon_map.get(ctype, "info-sign"),
            prefix="fa",
        ),
        tooltip=f"#{int(row['rank'])} {ctype} | {row.get('neighbourhood', '')} | {row['score']:.3f}",
    ).add_to(m)

# ─── Reference markers for Vincennes and Boulogne ───────────────────
folium.Marker(
    [48.8474, 2.4383],
    icon=folium.Icon(color="darkgreen", icon="map-pin", prefix="fa"),
    tooltip="Vincennes (reference point)",
    popup="<b>Vincennes</b><br>Check this area for opportunity gaps.<br>Toggle existing ATMs/FX to compare supply vs demand.",
).add_to(m)

folium.Marker(
    [48.8397, 2.2399],
    icon=folium.Icon(color="darkgreen", icon="map-pin", prefix="fa"),
    tooltip="Boulogne-Billancourt (reference point)",
    popup="<b>Boulogne-Billancourt</b><br>Check this area for opportunity gaps.<br>Toggle existing ATMs/FX to compare supply vs demand.",
).add_to(m)

# ─── Scored area boundary indicator ─────────────────────────────────
# Light dashed rectangle showing where the scoring grid was computed
folium.Rectangle(
    bounds=[[48.815, 2.22], [48.905, 2.47]],
    color="#95a5a6",
    weight=1,
    dash_array="8",
    fill=False,
    tooltip="Scoring grid boundary (analytical coverage)",
).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

# ─── Render ─────────────────────────────────────────────────────────
st.title("Paris ATM / FX Opportunity Explorer")

col_left, col_right = st.columns([3, 1])

with col_left:
    st_data = st_folium(m, width=1000, height=650, returned_objects=[])

with col_right:
    st.markdown("### Legend")
    st.markdown(f'<span style="color:{CANDIDATE_COLORS["ATM"]}">&#9679;</span> ATM candidates', unsafe_allow_html=True)
    st.markdown(f'<span style="color:{CANDIDATE_COLORS["FX Ad"]}">&#9679;</span> FX Ad candidates', unsafe_allow_html=True)
    st.markdown(f'<span style="color:{CANDIDATE_COLORS["FX Bureau"]}">&#9679;</span> FX Bureau candidates', unsafe_allow_html=True)
    st.markdown('<span style="color:#7f8c8d">&#9679;</span> Existing ATMs', unsafe_allow_html=True)
    st.markdown('<span style="color:#9b59b6">&#9679;</span> Existing FX bureaux', unsafe_allow_html=True)
    st.markdown('<span style="color:#3498db">&#9679;</span> Hotels', unsafe_allow_html=True)
    st.markdown('<span style="color:#2ecc71">&#9733;</span> Attractions', unsafe_allow_html=True)
    st.markdown('<span style="color:#f39c12">&#9632;</span> ZTI zones (orange overlay)', unsafe_allow_html=True)
    st.markdown('<span style="color:#2ecc71">&#9679;</span> Reference pins (Vincennes / Boulogne)', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Showing:** {len(filtered)} candidates")
    if len(filtered) > 0:
        st.markdown(f"**Score range:** {filtered['score'].min():.3f} - {filtered['score'].max():.3f}")

# ─── Candidate table ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Candidate Details")

if len(filtered) > 0:
    display_cols = ["type", "rank", "neighbourhood", "score",
                    "n_foreign", "n_airbnb", "n_hotels", "n_commercial",
                    "n_atm_scarcity", "n_fx_scarcity",
                    "atm_count_500m", "fx_bureau_count_500m",
                    "airbnb_count", "hotel_count",
                    "center_lat", "center_lon", "business_explanation"]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available].sort_values(["type", "rank"]),
        use_container_width=True,
        hide_index=True,
    )

    # Export
    csv = filtered[available].to_csv(index=False)
    st.download_button(
        "Download filtered candidates (CSV)",
        csv,
        "filtered_candidates.csv",
        "text/csv",
    )
else:
    st.info("No candidates match current filters. Adjust sidebar controls.")

# ─── Edge area analysis ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Vincennes & Boulogne Gap Check")
st.markdown("""
**How to inspect edge areas for opportunity gaps:**

1. Select **"Vincennes focus"** or **"Boulogne focus"** in the sidebar
2. Enable **"Existing ATMs"** and **"Existing FX bureaux"** layers
3. Enable **"Demand heatmap"** to see tourist/accommodation demand
4. Enable **"Hotels"** to see hotel coverage

If you see warm heatmap colors but few/no gray ATM dots or purple FX dots,
that area is **underserved** and may represent an opportunity gap.

**Note:** The analytical scoring grid covers the dashed rectangle on the map.
Areas outside this boundary (deep into Vincennes or Boulogne) were not scored
but can still be visually assessed by comparing demand vs supply layers.
""")
