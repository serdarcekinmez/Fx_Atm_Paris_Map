"""
Paris ATM / FX Placement Opportunity Analysis
==============================================
Identifies high-opportunity zones for ATM installation, FX advertising,
and potential FX bureau placement based on gap analysis between
tourist demand signals and existing supply coverage.
"""

import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
import folium
from folium.plugins import HeatMap

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────
GRID_SIZE = 0.002  # ~200m grid cells
RADIUS_M = 500     # radius for density calculations (meters)
PARIS_BOUNDS = (2.22, 48.815, 2.47, 48.905)  # lon_min, lat_min, lon_max, lat_max
CRS_WGS84 = 'EPSG:4326'
CRS_METRIC = 'EPSG:2154'  # Lambert 93 for distance calculations

# ─── 1. Data Loading ────────────────────────────────────────────────

def load_data():
    """Load all datasets and harmonize to WGS84."""
    print("Loading datasets...")

    # Airbnb / FX target data
    fx = pd.read_csv('processed_fx_target_data.csv')
    fx_gdf = gpd.GeoDataFrame(
        fx, geometry=gpd.points_from_xy(fx.longitude, fx.latitude), crs=CRS_WGS84
    )

    # DATAtourisme
    dt = pd.read_csv('datatourisme.csv', encoding='utf-8')
    dt_gdf = gpd.GeoDataFrame(
        dt, geometry=gpd.points_from_xy(dt.Longitude, dt.Latitude), crs=CRS_WGS84
    )

    # GeoJSON files
    atm = gpd.read_file('atm_paris.geojson').to_crs(CRS_WGS84)
    fx_bur = gpd.read_file('fx_bureaux.geojson').to_crs(CRS_WGS84)
    hotels = gpd.read_file('hotels.geojson').to_crs(CRS_WGS84)
    commercial = gpd.read_file('Local_commercial_par_taille_(6_postes).geojson').to_crs(CRS_WGS84)

    # Touristical attractions - handle encoding
    with open('touristical_attraction.geojson', 'r', encoding='utf-8') as f:
        ta_data = json.load(f)
    attractions = gpd.GeoDataFrame.from_features(ta_data['features'], crs=CRS_WGS84)

    zti = gpd.read_file('zones-touristiques-internationales.geojson').to_crs(CRS_WGS84)

    print(f"  FX target: {len(fx_gdf)} | Hotels: {len(hotels)} | ATMs: {len(atm)}")
    print(f"  FX bureaux: {len(fx_bur)} | Attractions: {len(attractions)} | DATAtourisme: {len(dt_gdf)}")
    print(f"  Commercial locals: {len(commercial)} | ZTI zones: {len(zti)}")

    return fx_gdf, dt_gdf, atm, fx_bur, hotels, commercial, attractions, zti


# ─── 2. Grid Construction ───────────────────────────────────────────

def build_grid():
    """Create analysis grid over Paris."""
    print("Building analysis grid...")
    lons = np.arange(PARIS_BOUNDS[0], PARIS_BOUNDS[2], GRID_SIZE)
    lats = np.arange(PARIS_BOUNDS[1], PARIS_BOUNDS[3], GRID_SIZE)

    cells = []
    centroids = []
    for lon in lons:
        for lat in lats:
            cell = box(lon, lat, lon + GRID_SIZE, lat + GRID_SIZE)
            cells.append(cell)
            centroids.append(Point(lon + GRID_SIZE/2, lat + GRID_SIZE/2))

    grid = gpd.GeoDataFrame({
        'geometry': cells,
        'centroid_geom': centroids,
        'center_lon': [c.x for c in centroids],
        'center_lat': [c.y for c in centroids],
    }, crs=CRS_WGS84)

    print(f"  Grid cells: {len(grid)}")
    return grid


# ─── 3. Density Calculation Helpers ─────────────────────────────────

def count_points_in_radius(grid_points_metric, source_points_metric, radius=RADIUS_M):
    """Count source points within radius of each grid centroid using KDTree."""
    if len(source_points_metric) == 0:
        return np.zeros(len(grid_points_metric))

    tree = cKDTree(source_points_metric)
    counts = tree.query_ball_point(grid_points_metric, r=radius)
    return np.array([len(c) for c in counts])


def nearest_distance(grid_points_metric, source_points_metric):
    """Distance to nearest source point from each grid centroid."""
    if len(source_points_metric) == 0:
        return np.full(len(grid_points_metric), 5000.0)  # default large distance

    tree = cKDTree(source_points_metric)
    distances, _ = tree.query(grid_points_metric, k=1)
    return distances


def get_metric_coords(gdf):
    """Extract metric coordinates as numpy array."""
    gdf_m = gdf.to_crs(CRS_METRIC)
    return np.column_stack([gdf_m.geometry.x, gdf_m.geometry.y])


# ─── 4. Demand Signal Construction ──────────────────────────────────

def build_demand_signals(grid, fx_gdf, hotels, attractions, dt_gdf, zti):
    """Compute demand-side scores for each grid cell."""
    print("Building demand signals...")

    # Convert grid centroids to metric
    grid_centroids = gpd.GeoDataFrame(
        geometry=grid['centroid_geom'].values, crs=CRS_WGS84
    ).to_crs(CRS_METRIC)
    grid_pts = np.column_stack([grid_centroids.geometry.x, grid_centroids.geometry.y])

    # --- Airbnb / FX target signals ---
    fx_m = get_metric_coords(fx_gdf)

    # Raw density
    grid['airbnb_count'] = count_points_in_radius(grid_pts, fx_m)

    # Weighted foreign demand: aggregate foreign_tourist_ratio weighted by reviews
    # For each grid cell, find nearby listings and compute weighted avg
    fx_gdf_m = fx_gdf.to_crs(CRS_METRIC)
    fx_coords = np.column_stack([fx_gdf_m.geometry.x, fx_gdf_m.geometry.y])

    # Use tree to find listings in radius for weighted calcs
    fx_tree = cKDTree(fx_coords)

    foreign_ratios = fx_gdf['foreign_tourist_ratio_pct'].fillna(0).values
    occupancy = fx_gdf['occupancy_rate_pct'].fillna(0).values
    reviews = fx_gdf['number_of_reviews'].fillna(0).values

    weighted_foreign = np.zeros(len(grid))
    weighted_occupancy = np.zeros(len(grid))
    weighted_review_volume = np.zeros(len(grid))

    neighbors_list = fx_tree.query_ball_point(grid_pts, r=RADIUS_M)

    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) > 0:
            idx = np.array(neighbors)
            rev = reviews[idx]
            total_rev = rev.sum()
            if total_rev > 0:
                weighted_foreign[i] = np.average(foreign_ratios[idx], weights=rev + 1)
                weighted_occupancy[i] = np.average(occupancy[idx], weights=rev + 1)
            else:
                weighted_foreign[i] = foreign_ratios[idx].mean()
                weighted_occupancy[i] = occupancy[idx].mean()
            weighted_review_volume[i] = total_rev

    grid['foreign_demand'] = weighted_foreign
    grid['occupancy_signal'] = weighted_occupancy
    grid['review_volume'] = weighted_review_volume

    # --- Hotel density ---
    hotel_m = get_metric_coords(hotels)
    grid['hotel_count'] = count_points_in_radius(grid_pts, hotel_m)

    # --- Tourist attraction density ---
    attr_m = get_metric_coords(attractions)
    grid['attraction_count'] = count_points_in_radius(grid_pts, attr_m)
    grid['attraction_dist'] = nearest_distance(grid_pts, attr_m)

    # --- DATAtourisme density ---
    dt_m = get_metric_coords(dt_gdf)
    grid['datatourisme_count'] = count_points_in_radius(grid_pts, dt_m)

    # --- ZTI proximity ---
    # Check if grid centroid falls within any ZTI zone
    grid_centroids_wgs = gpd.GeoDataFrame(geometry=grid['centroid_geom'].values, crs=CRS_WGS84)
    zti_union = zti.union_all()
    grid['in_zti'] = grid_centroids_wgs.geometry.within(zti_union).astype(int)

    # Also compute distance to nearest ZTI boundary for near-ZTI boost
    zti_m = zti.to_crs(CRS_METRIC)
    zti_boundary = zti_m.union_all().boundary
    grid['zti_dist'] = grid_centroids.geometry.distance(zti_boundary)
    grid['zti_proximity_boost'] = np.clip(1 - grid['zti_dist'] / 1000, 0, 1)  # boost within 1km
    grid.loc[grid['in_zti'] == 1, 'zti_proximity_boost'] = 1.0

    print("  Demand signals complete.")
    return grid


# ─── 5. Supply / Coverage Signals ───────────────────────────────────

def build_supply_signals(grid, atm, fx_bur, commercial):
    """Compute supply-side and commercial viability scores."""
    print("Building supply signals...")

    grid_centroids = gpd.GeoDataFrame(
        geometry=grid['centroid_geom'].values, crs=CRS_WGS84
    ).to_crs(CRS_METRIC)
    grid_pts = np.column_stack([grid_centroids.geometry.x, grid_centroids.geometry.y])

    # ATM coverage
    atm_m = get_metric_coords(atm)
    grid['atm_count_500m'] = count_points_in_radius(grid_pts, atm_m, RADIUS_M)
    grid['atm_nearest_dist'] = nearest_distance(grid_pts, atm_m)

    # FX bureau coverage
    fx_m = get_metric_coords(fx_bur)
    grid['fx_bureau_count_500m'] = count_points_in_radius(grid_pts, fx_m, RADIUS_M)
    grid['fx_nearest_dist'] = nearest_distance(grid_pts, fx_m)

    # Commercial local density (viability proxy)
    comm_m = get_metric_coords(commercial)
    grid['commercial_count'] = count_points_in_radius(grid_pts, comm_m, RADIUS_M)

    print("  Supply signals complete.")
    return grid


# ─── 6. Scoring Model ───────────────────────────────────────────────

def normalize_col(series, clip_percentile=99):
    """Min-max normalize with outlier clipping."""
    upper = np.percentile(series, clip_percentile)
    clipped = np.clip(series, 0, upper)
    rng = clipped.max() - clipped.min()
    if rng == 0:
        return np.zeros_like(series)
    return (clipped - clipped.min()) / rng


def score_candidates(grid):
    """
    Scoring formula:

    ATM_SCORE = (
        0.20 * airbnb_density_norm +
        0.20 * foreign_demand_norm +
        0.10 * occupancy_norm +
        0.15 * hotel_density_norm +
        0.10 * attraction_proximity_norm +
        0.05 * datatourisme_norm +
        0.05 * zti_boost +
        0.05 * commercial_viability_norm +
        0.10 * atm_scarcity_norm           ← INVERSE: fewer ATMs = higher score
    )

    FX_AD_SCORE = (
        0.15 * airbnb_density_norm +
        0.25 * foreign_demand_norm +        ← FX ads weight foreign tourists more
        0.05 * occupancy_norm +
        0.15 * hotel_density_norm +
        0.05 * attraction_proximity_norm +
        0.05 * datatourisme_norm +
        0.10 * zti_boost +                  ← ZTI matters more for FX ads
        0.10 * commercial_viability_norm +   ← visibility matters for ads
        0.05 * fx_scarcity_norm +
        0.05 * review_volume_norm
    )
    """
    print("Scoring candidates...")

    # Normalize demand signals
    grid['n_airbnb'] = normalize_col(grid['airbnb_count'])
    grid['n_foreign'] = normalize_col(grid['foreign_demand'])
    grid['n_occupancy'] = normalize_col(grid['occupancy_signal'])
    grid['n_hotels'] = normalize_col(grid['hotel_count'])
    grid['n_attractions'] = normalize_col(grid['attraction_count'])
    grid['n_datatourisme'] = normalize_col(grid['datatourisme_count'])
    grid['n_zti'] = grid['zti_proximity_boost']  # already 0-1
    grid['n_commercial'] = normalize_col(grid['commercial_count'])
    grid['n_review_vol'] = normalize_col(grid['review_volume'])

    # Scarcity signals (inverse: far from ATM / few ATMs = high scarcity = high opportunity)
    # ATM scarcity: combine distance and inverse density
    grid['atm_scarcity'] = (
        normalize_col(grid['atm_nearest_dist']) * 0.6 +
        normalize_col(grid['atm_count_500m'].max() - grid['atm_count_500m']) * 0.4
    )
    grid['n_atm_scarcity'] = normalize_col(grid['atm_scarcity'])

    # FX scarcity
    grid['fx_scarcity'] = (
        normalize_col(grid['fx_nearest_dist']) * 0.6 +
        normalize_col(grid['fx_bureau_count_500m'].max() - grid['fx_bureau_count_500m']) * 0.4
    )
    grid['n_fx_scarcity'] = normalize_col(grid['fx_scarcity'])

    # ── Hard saturation penalty for ATMs ──
    # If >8 ATMs within 500m, apply a penalty multiplier
    grid['atm_saturation_penalty'] = np.where(
        grid['atm_count_500m'] > 10, 0.5,
        np.where(grid['atm_count_500m'] > 6, 0.75, 1.0)
    )

    # ── ATM Opportunity Score ──
    grid['atm_score_raw'] = (
        0.18 * grid['n_airbnb'] +
        0.18 * grid['n_foreign'] +
        0.08 * grid['n_occupancy'] +
        0.12 * grid['n_hotels'] +
        0.08 * grid['n_attractions'] +
        0.04 * grid['n_datatourisme'] +
        0.05 * grid['n_zti'] +
        0.05 * grid['n_commercial'] +
        0.22 * grid['n_atm_scarcity']     # increased from 0.10 to 0.22
    )
    grid['atm_score'] = grid['atm_score_raw'] * grid['atm_saturation_penalty']

    # ── FX Ad Opportunity Score ──
    grid['fx_ad_score'] = (
        0.15 * grid['n_airbnb'] +
        0.25 * grid['n_foreign'] +
        0.05 * grid['n_occupancy'] +
        0.15 * grid['n_hotels'] +
        0.05 * grid['n_attractions'] +
        0.05 * grid['n_datatourisme'] +
        0.10 * grid['n_zti'] +
        0.10 * grid['n_commercial'] +
        0.05 * grid['n_fx_scarcity'] +
        0.05 * grid['n_review_vol']
    )

    # ── FX Bureau Score (more selective) ──
    # Requires: high foreign demand + high volume + commercial presence + FX scarcity
    grid['fx_bureau_score'] = (
        0.10 * grid['n_airbnb'] +
        0.30 * grid['n_foreign'] +
        0.10 * grid['n_occupancy'] +
        0.10 * grid['n_hotels'] +
        0.05 * grid['n_attractions'] +
        0.05 * grid['n_zti'] +
        0.15 * grid['n_commercial'] +
        0.10 * grid['n_fx_scarcity'] +
        0.05 * grid['n_review_vol']
    )

    # Filter out cells with zero demand (outside Paris proper or no data)
    grid['has_demand'] = (grid['airbnb_count'] > 0) | (grid['hotel_count'] > 0)

    print(f"  Scored {grid['has_demand'].sum()} active grid cells.")
    return grid


# ─── 7. Arrondissement Labeling ─────────────────────────────────────

def label_arrondissements(grid, fx_gdf):
    """Assign approximate arrondissement from nearest Airbnb listing."""
    print("Labeling arrondissements...")

    # Build mapping from fx_gdf neighbourhoods
    fx_m = fx_gdf.to_crs(CRS_METRIC)
    fx_coords = np.column_stack([fx_m.geometry.x, fx_m.geometry.y])
    fx_tree = cKDTree(fx_coords)

    grid_centroids = gpd.GeoDataFrame(
        geometry=grid['centroid_geom'].values, crs=CRS_WGS84
    ).to_crs(CRS_METRIC)
    grid_pts = np.column_stack([grid_centroids.geometry.x, grid_centroids.geometry.y])

    _, nearest_idx = fx_tree.query(grid_pts, k=1)
    grid['neighbourhood'] = fx_gdf['neighbourhood_cleansed'].iloc[nearest_idx].values

    return grid


# ─── 8. Export Results ───────────────────────────────────────────────

def extract_top_candidates(grid, score_col, n=15, label=''):
    """Extract top N non-overlapping candidate zones."""
    active = grid[grid['has_demand']].copy()
    active = active.sort_values(score_col, ascending=False)

    # Deduplicate: ensure candidates are at least 300m apart
    selected = []
    used_coords = []

    for _, row in active.iterrows():
        if len(selected) >= n:
            break

        coord = (row['center_lon'], row['center_lat'])
        too_close = False
        for uc in used_coords:
            dist = ((coord[0] - uc[0])**2 + (coord[1] - uc[1])**2)**0.5
            if dist < 0.003:  # ~300m
                too_close = True
                break

        if not too_close:
            selected.append(row)
            used_coords.append(coord)

    result = pd.DataFrame(selected)
    result['rank'] = range(1, len(result) + 1)

    cols = ['rank', 'neighbourhood', 'center_lat', 'center_lon', score_col,
            'n_foreign', 'n_airbnb', 'n_hotels', 'n_attractions', 'n_zti',
            'n_commercial', 'n_atm_scarcity', 'n_fx_scarcity',
            'airbnb_count', 'hotel_count', 'atm_count_500m', 'fx_bureau_count_500m']

    available_cols = [c for c in cols if c in result.columns]
    return result[available_cols]


def make_business_explanation(row, score_type):
    """Generate short business rationale for a candidate."""
    parts = []

    if row.get('n_foreign', 0) > 0.6:
        parts.append("high foreign tourist ratio")
    if row.get('n_airbnb', 0) > 0.5:
        parts.append("dense short-stay accommodation")
    if row.get('n_hotels', 0) > 0.5:
        parts.append("strong hotel cluster")
    if row.get('n_zti', 0) > 0.5:
        parts.append("in/near international tourism zone")
    if row.get('n_attractions', 0) > 0.3:
        parts.append("near tourist attractions")
    if row.get('n_commercial', 0) > 0.5:
        parts.append("active commercial street")

    if score_type == 'atm':
        if row.get('n_atm_scarcity', 0) > 0.5:
            parts.append("ATM coverage gap")
        if row.get('atm_count_500m', 0) <= 2:
            parts.append(f"only {int(row.get('atm_count_500m', 0))} ATMs within 500m")
    elif score_type == 'fx':
        if row.get('n_fx_scarcity', 0) > 0.5:
            parts.append("FX bureau gap")

    return "; ".join(parts) if parts else "multi-signal opportunity"


def create_map(grid, candidates, score_col, title, filename, atm=None, fx_bur=None):
    """Create an interactive Folium map."""
    print(f"  Creating map: {filename}")

    center = [48.8566, 2.3522]
    m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')

    # Heatmap layer from grid scores
    active = grid[grid['has_demand']].copy()
    heat_data = active[['center_lat', 'center_lon', score_col]].values.tolist()
    HeatMap(heat_data, radius=15, blur=20, max_zoom=15, name=f'{title} Heatmap').add_to(m)

    # Candidate markers
    candidate_group = folium.FeatureGroup(name='Top Candidates')
    for _, row in candidates.iterrows():
        explanation = make_business_explanation(row, 'atm' if 'atm' in score_col else 'fx')
        popup_text = f"""
        <b>Rank #{int(row['rank'])}</b><br>
        <b>{row.get('neighbourhood', 'N/A')}</b><br>
        Score: {row[score_col]:.3f}<br>
        Foreign demand: {row.get('n_foreign', 0):.2f}<br>
        Airbnb density: {row.get('n_airbnb', 0):.2f}<br>
        Hotels nearby: {int(row.get('hotel_count', 0))}<br>
        ATMs in 500m: {int(row.get('atm_count_500m', 0))}<br>
        FX bureaux in 500m: {int(row.get('fx_bureau_count_500m', 0))}<br>
        <i>{explanation}</i>
        """

        color = 'red' if row['rank'] <= 5 else ('orange' if row['rank'] <= 10 else 'blue')
        folium.Marker(
            [row['center_lat'], row['center_lon']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=color, icon='info-sign'),
            tooltip=f"#{int(row['rank'])} {row.get('neighbourhood', '')}"
        ).add_to(candidate_group)
    candidate_group.add_to(m)

    # Existing ATMs (small dots)
    if atm is not None:
        atm_group = folium.FeatureGroup(name='Existing ATMs', show=False)
        for _, row in atm.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=3, color='gray', fill=True, fill_opacity=0.5,
                tooltip=row.get('name', 'ATM')
            ).add_to(atm_group)
        atm_group.add_to(m)

    # Existing FX bureaux
    if fx_bur is not None:
        fx_group = folium.FeatureGroup(name='Existing FX Bureaux', show=False)
        for _, row in fx_bur.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=4, color='purple', fill=True, fill_opacity=0.7,
                tooltip=row.get('name', 'FX Bureau')
            ).add_to(fx_group)
        fx_group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(filename)
    print(f"  Saved: {filename}")


def export_csv(candidates, score_col, score_type, filename):
    """Export ranked candidates to CSV."""
    df = candidates.copy()
    df['business_explanation'] = df.apply(
        lambda r: make_business_explanation(r, score_type), axis=1
    )
    df.to_csv(filename, index=False)
    print(f"  Saved: {filename}")


# ─── 9. Challenge / Sensitivity Pass ────────────────────────────────

def challenge_results(grid, atm_cands, fx_cands):
    """Validate results against common biases."""
    print("\n--- Challenge Pass ---")

    # Check 1: Are top results all in the same arrondissement?
    atm_hoods = atm_cands['neighbourhood'].value_counts()
    print(f"  ATM candidates span {len(atm_hoods)} neighbourhoods:")
    print(f"    {atm_hoods.to_dict()}")

    fx_hoods = fx_cands['neighbourhood'].value_counts()
    print(f"  FX ad candidates span {len(fx_hoods)} neighbourhoods:")
    print(f"    {fx_hoods.to_dict()}")

    # Check 2: Are any top candidates already saturated?
    atm_saturated = atm_cands[atm_cands['atm_count_500m'] > 8]
    if len(atm_saturated) > 0:
        print(f"  WARNING: {len(atm_saturated)} ATM candidates have >8 ATMs within 500m")
    else:
        print("  OK: No ATM candidates are heavily saturated")

    # Check 3: Do any candidates have zero demand signals?
    low_demand = atm_cands[atm_cands['airbnb_count'] < 5]
    if len(low_demand) > 0:
        print(f"  WARNING: {len(low_demand)} ATM candidates have <5 Airbnb listings nearby")
    else:
        print("  OK: All ATM candidates have meaningful demand")

    print("--- Challenge Pass Complete ---\n")


# ─── 10. Main Pipeline ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PARIS ATM / FX OPPORTUNITY ANALYSIS")
    print("=" * 60)

    # Load
    fx_gdf, dt_gdf, atm, fx_bur, hotels, commercial, attractions, zti = load_data()

    # Grid
    grid = build_grid()

    # Signals
    grid = build_demand_signals(grid, fx_gdf, hotels, attractions, dt_gdf, zti)
    grid = build_supply_signals(grid, atm, fx_bur, commercial)

    # Score
    grid = score_candidates(grid)
    grid = label_arrondissements(grid, fx_gdf)

    # Extract top candidates
    print("\nExtracting top candidates...")
    atm_cands = extract_top_candidates(grid, 'atm_score', n=15)
    fx_cands = extract_top_candidates(grid, 'fx_ad_score', n=15)
    fx_bureau_cands = extract_top_candidates(grid, 'fx_bureau_score', n=10)

    # Challenge
    challenge_results(grid, atm_cands, fx_cands)

    # Export
    print("Exporting results...")
    export_csv(atm_cands, 'atm_score', 'atm', 'atm_candidates.csv')
    export_csv(fx_cands, 'fx_ad_score', 'fx', 'fx_ad_candidates.csv')
    export_csv(fx_bureau_cands, 'fx_bureau_score', 'fx', 'fx_bureau_candidates.csv')

    create_map(grid, atm_cands, 'atm_score', 'ATM Opportunity',
               'paris_atm_opportunity_map.html', atm=atm, fx_bur=fx_bur)
    create_map(grid, fx_cands, 'fx_ad_score', 'FX Ad Opportunity',
               'paris_fx_ad_opportunity_map.html', atm=atm, fx_bur=fx_bur)

    # Print summary tables
    print("\n" + "=" * 60)
    print("TOP 15 ATM PLACEMENT CANDIDATES")
    print("=" * 60)
    for _, r in atm_cands.iterrows():
        expl = make_business_explanation(r, 'atm')
        print(f"  #{int(r['rank']):2d} | {r.get('neighbourhood',''):25s} | "
              f"({r['center_lat']:.4f}, {r['center_lon']:.4f}) | "
              f"Score: {r['atm_score']:.3f} | "
              f"ATMs nearby: {int(r.get('atm_count_500m',0))} | {expl}")

    print("\n" + "=" * 60)
    print("TOP 15 FX ADVERTISING CANDIDATES")
    print("=" * 60)
    for _, r in fx_cands.iterrows():
        expl = make_business_explanation(r, 'fx')
        print(f"  #{int(r['rank']):2d} | {r.get('neighbourhood',''):25s} | "
              f"({r['center_lat']:.4f}, {r['center_lon']:.4f}) | "
              f"Score: {r['fx_ad_score']:.3f} | "
              f"FX nearby: {int(r.get('fx_bureau_count_500m',0))} | {expl}")

    print("\n" + "=" * 60)
    print("TOP 10 FX BUREAU CANDIDATES")
    print("=" * 60)
    for _, r in fx_bureau_cands.iterrows():
        expl = make_business_explanation(r, 'fx')
        print(f"  #{int(r['rank']):2d} | {r.get('neighbourhood',''):25s} | "
              f"({r['center_lat']:.4f}, {r['center_lon']:.4f}) | "
              f"Score: {r['fx_bureau_score']:.3f} | {expl}")

    print("\n[OK] Analysis complete. Files exported.")
    print("  - atm_candidates.csv")
    print("  - fx_ad_candidates.csv")
    print("  - fx_bureau_candidates.csv")
    print("  - paris_atm_opportunity_map.html")
    print("  - paris_fx_ad_opportunity_map.html")


if __name__ == '__main__':
    main()
