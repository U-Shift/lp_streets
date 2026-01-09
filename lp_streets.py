# Importing necessary libraries
from branca.colormap import LinearColormap as BrancaLinearColormap
from branca.element import Element
import builtins
import contextily as ctx
import folium
import geopandas as gpd
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import neatnet
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import CRS
import re
from scipy.stats import pearsonr
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from sklearn.decomposition import PCA

# -----------------------------------

link_palette_cat = {
        "V":   "#ffffa3",
        "IV":  "#ffa84b",
        "III": "#ff0000",
        "II":  "#bf0000",
        "I":   "#580000"
}

link_palette_grad = [
               "#ffffa3",
               "#ff0000",
               "#550000"
]

place_palette_cat = {
        "E":   "#ffffa3",
        "D":   "#bfff5a",
        "C":   "#00ff00",
        "B":   "#00a800",
        "A":   "#0B3800"
}

place_palette_grad = [
               "#ffffa3",
               "#00ff00",
               "#0A3000"
]

full_palette_cat = {
    "I-E":   "#ff0000", "I-D":   "#e40000", "I-C":   "#bf0000", "I-B":   "#8a0000", "I-A":   "#332E00",
    "II-E":  "#ff722e", "II-D":  "#e47222", "II-C":  "#bf7219", "II-B":  "#8a7211", "II-A":  "#007200",
    "III-E": "#ffa84b", "III-D": "#e4a838", "III-C": "#bfa829", "III-B": "#8aa81b", "III-A": "#00a800",
    "IV-E":  "#ffd66b", "IV-D":  "#e4d64f", "IV-C":  "#bfd63b", "IV-B":  "#8ad626", "IV-A":  "#00d600",
    "V-E":   "#ffffa3", "V-D":   "#e4ff79", "V-C":   "#bfff5a", "V-B":   "#8aff3a", "V-A":   "#00ff00"
}

# -----------------------------------

study_area = "Município de Lisboa, Portugal"

local_CRS = "epsg:3763"

speed_limits = {
    "motorway": 120,
    "motorway_link": 60,
    "trunk": 100,
    "trunk_link": 60,
    "primary": 50,
    "primary_link": 50,
    "secondary": 50,
    "secondary_link": 50,
    "tertiary": 50,
    "tertiary_link": 50,
    "residential": 50,
    "unclassified": 90,
    "living_street": 20,
    "pedestrian": 0
}

# -----------------------------------

study_area_gdf = ox.geocode_to_gdf(study_area)
study_area_gdf = study_area_gdf.to_crs(local_CRS)
study_area_3857 = study_area_gdf.to_crs(epsg=3857)

def filter_ground_level(gdf):
    # Ensure 'layer' column exists
    if "layer" not in gdf.columns:
        gdf["layer"] = "0"
    layer_num = pd.to_numeric(gdf["layer"], errors="coerce").fillna(0)
    return gdf[layer_num >= 0]

# Retrieving buildings
buildings = ox.features_from_place(study_area, tags={"building": True})

# Applying filters to the "building" features
if "building" in buildings.columns:
    buildings = buildings[buildings["building"] != "roof"]
    buildings = buildings[buildings["building"] != "container"]
    buildings = buildings[buildings["building"] != "kiosk"]
    buildings = buildings[buildings["building"] != "memorial"]
    buildings = buildings[buildings["building"] != "service"]
    buildings = buildings[buildings["building"] != "guardhouse"]
    buildings = buildings[buildings["building"] != "train_station"]

if "amenity" in buildings.columns:
    buildings = buildings[buildings["amenity"] != "shelter"]
    buildings = buildings[buildings["amenity"] != "fountain"]
    buildings = buildings[buildings["amenity"] != "toilets"]

if "artwork_type" in buildings.columns:
    buildings = buildings[buildings["artwork_type"] != "statue"]

if "historic" in buildings.columns:
    buildings = buildings[buildings["historic"] != "monument"]
    buildings = buildings[buildings["historic"] != "memorial"]

if "memorial" in buildings.columns:
    buildings = buildings[buildings["memorial"] != "statue"]
    buildings = buildings[buildings["memorial"] != "bust"]

if "shop" in buildings.columns:
    buildings = buildings[buildings["shop"] != "kiosk"]

if "bridge:support" in buildings.columns:
    buildings = buildings[buildings["bridge:support"] != "yes"]
    buildings = buildings[buildings["bridge:support"] != "pier"]
    buildings = buildings[buildings["bridge:support"] != "abutment"]
    buildings = buildings[buildings["bridge:support"] != "lift_pier"]
    buildings = buildings[buildings["bridge:support"] != "pivot_pier"]
    buildings = buildings[buildings["bridge:support"] != "pylon"]
    
# Filtering for ground level features
buildings = filter_ground_level(buildings)

# Retrieving construction areas
construction = ox.features_from_place(study_area, tags={"landuse": "construction"})

# Filtering for ground level features
construction = filter_ground_level(construction)

# Retrieving schools
schools = ox.features_from_place(study_area, tags={"amenity": "school"})

# Filtering for ground level features
schools = filter_ground_level(schools)

# Retrieving pitches
pitches = ox.features_from_place(study_area, tags={"leisure": "pitch"})

# Filtering for ground level features
pitches = filter_ground_level(pitches)

# Retrieving cemeteries
cemeteries = ox.features_from_place(study_area, tags={"landuse": "cemetery"})

# Filtering for ground level features
cemeteries = filter_ground_level(cemeteries)

buildings = buildings.to_crs(local_CRS)
construction = construction.to_crs(local_CRS)
schools = schools.to_crs(local_CRS)
pitches = pitches.to_crs(local_CRS)
cemeteries = cemeteries.to_crs(local_CRS)

exclusion_mask = gpd.GeoDataFrame(
    pd.concat([
       buildings[['geometry']],
       construction[['geometry']],
       schools[['geometry']],
       pitches[['geometry']],
       cemeteries[['geometry']]    
    ], ignore_index = True)
)

exclusion_mask = gpd.GeoSeries(unary_union(exclusion_mask.geometry), crs=local_CRS)

# Filtering for wanted highway types
cf_highway_types = [
    'motorway',
    'motorway_link',
    'trunk',
    'trunk_link',
    'primary',
    'primary_link',
    'secondary',
    'secondary_link',
    'tertiary',
    'tertiary_link',
    'residential',
    'unclassified',
    'living_street',
    'pedestrian'
]

cf = '["highway"~"{}"]'.format('|'.join(cf_highway_types))

# Filtering out area highway types
cf += cf + '["area"!~"yes"]'
cf += cf + '["area:highway"!~"footway"]'
cf += cf + '["area:highway"!~"path"]'
cf += cf + '["area:highway"!~"steps"]'
cf += cf + '["area:highway"!~"pedestrian"]'

# Extend OSMnx useful tags so edges carry directional and PSV info
extra_way_tags = [
    # directional lane counts
    "lanes:forward", "lanes:backward",
    # PSV numeric counts
    "lanes:psv", "lanes:psv:forward", "lanes:psv:backward",
    # PSV per-lane designation strings
    "psv:lanes", "psv:lanes:forward", "psv:lanes:backward",
    "bus:lanes", "bus:lanes:forward", "bus:lanes:backward",  # sometimes used instead of psv
    # tunnels and bridges
    "tunnel", "bridge"
]

ox.settings.useful_tags_way = sorted(set(list(ox.settings.useful_tags_way) + extra_way_tags))

network = ox.graph_from_place(
    study_area,
    custom_filter=cf,
    retain_all=False, 
    simplify=False, 
    truncate_by_edge=True
)

network_gdf = ox.graph_to_gdfs(network, nodes=False, edges=True)

network_gdf = network_gdf[network_gdf.geometry.notnull()]

network_gdf = network_gdf[network_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]

network_gdf = network_gdf.to_crs(local_CRS)

# Defining roundabouts as "oneway"=True, if they're empty
network_gdf.loc[network_gdf["junction"].isin(["roundabout"]) & network_gdf["oneway"].isnull(), "oneway"] = True

# Defining "motorway", "motorway_link", "trunk", and "trunk_link" as "oneway"=True, if they're empty
network_gdf.loc[network_gdf["highway"].isin(["motorway", "motorway_link", "trunk", "trunk_link"]) & network_gdf["oneway"].isnull(), "oneway"] = True

# Defining remaining highways as "oneway"= False, if they're empty
network_gdf.loc[network_gdf["oneway"].isnull(), "oneway"] = False

network_gdf.loc[
    (network_gdf["lanes"].isnull()) & 
    (network_gdf["highway"].isin(["residential", "unclassified", "living_street"])) & 
    (network_gdf["oneway"] == True), 
    "lanes"
] = 1

network_gdf.loc[
    (network_gdf["lanes"].isnull()) & 
    (network_gdf["highway"].isin(["residential", "unclassified", "living_street"])) &
    (network_gdf["oneway"] == False), 
    "lanes"
] = 2

# Calculate weighted average of "lanes" for each "highway" type
# Only use rows where "lanes" is not null and "length" is available
valid_lanes = network_gdf[network_gdf["lanes"].notnull() & network_gdf["length"].notnull()].copy()
valid_lanes["lanes"] = pd.to_numeric(valid_lanes["lanes"], errors="coerce")

weighted_avg_lanes = (
    valid_lanes.groupby("highway")[["lanes", "length"]]
    .apply(lambda df: np.average(df["lanes"], weights=df["length"]))
    .round()
    .astype(int)
)

# Fill missing "lanes" values using the rounded weighted average for each "highway"
def fill_lanes(row):
    if pd.isnull(row["lanes"]):
        return weighted_avg_lanes.get(row["highway"], np.nan)
    return row["lanes"]

network_gdf["lanes"] = network_gdf.apply(fill_lanes, axis=1)

# 1) General lane counts
# Ensure total lanes is numeric to avoid string - float errors
lanes_tot = pd.to_numeric(network_gdf.get("lanes"), errors="coerce")

lf = pd.to_numeric(network_gdf.get("lanes:forward"), errors="coerce")
lb = pd.to_numeric(network_gdf.get("lanes:backward"), errors="coerce")

# Start with explicit directional tags if present
lanes_forward_dir = lf.copy()
lanes_backward_dir = lb.copy()

# Where missing, derive from oneway and total lanes
mask_missing_both = lanes_forward_dir.isna() & lanes_backward_dir.isna()
if mask_missing_both.any():
    # oneway -> all lanes in the forward direction
    oneway_mask = mask_missing_both & (network_gdf["oneway"] == True)
    lanes_forward_dir.loc[oneway_mask]  = lanes_tot.loc[oneway_mask]
    lanes_backward_dir.loc[oneway_mask] = 0

    # two-way -> split total lanes
    tw_mask = mask_missing_both & (network_gdf["oneway"] == False)
    tot = lanes_tot.loc[tw_mask].fillna(0).astype(float)
    split_f = np.floor(tot / 2.0).astype(int)
    split_b = (tot - split_f).astype(int)
    lanes_forward_dir.loc[tw_mask]  = split_f
    lanes_backward_dir.loc[tw_mask] = split_b

# Any remaining single-side NaNs: backfill by difference with total
rem_f = lanes_forward_dir.isna() & lanes_tot.notna()
lanes_forward_dir.loc[rem_f] = (lanes_tot.loc[rem_f] - lanes_backward_dir.loc[rem_f].fillna(0)).clip(lower=0)

rem_b = lanes_backward_dir.isna() & lanes_tot.notna()
lanes_backward_dir.loc[rem_b] = (lanes_tot.loc[rem_b] - lanes_forward_dir.loc[rem_b].fillna(0)).clip(lower=0)

# Final integer, nonnegative
lanes_forward_dir  = lanes_forward_dir.fillna(0).astype(int).clip(lower=0)
lanes_backward_dir = lanes_backward_dir.fillna(0).astype(int).clip(lower=0)

def _to_int(x):
    try:
        if pd.isna(x):
            return np.nan

        # handle lists/tuples: pick first non-null element
        if isinstance(x, (list, tuple)):
            for e in x:
                if not pd.isna(e):
                    x = e
                    break
            else:
                return np.nan

        # strings: try several sane parsing strategies
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return np.nan
            # normalize separators to pipe
            s = re.sub(r"[;,/]+", "|", s)

            # if pipe-delimited, prefer the first numeric token
            if "|" in s:
                toks = [t.strip() for t in s.split("|") if t.strip() != ""]
                for t in toks:
                    if re.match(r"^[-+]?[0-9]+(\\.[0-9]+)?$", t):
                        return int(float(t))
                # fall through to try parsing first token
                s = toks[0]

            # ranges like 2-3 -> take min(2,3)
            m = re.match(r"^(?P<a>[-+]?[0-9]+(\\.[0-9]+)?)\\s*[-–]\\s*(?P<b>[-+]?[0-9]+(\\.[0-9]+)?)$", s)
            if m:
                a = float(m.group("a"))
                b = float(m.group("b"))
                return int(min(a, b))

            # extract first numeric occurrence (handles "50 km/h", "50mph")
            m = re.search(r"([-+]?[0-9]+(\\.[0-9]+)?)", s)
            if m:
                return int(float(m.group(1)))

            return np.nan

        # numeric types
        return int(float(x))
    except Exception:
        return np.nan

def _count_psv_from_token_string(s):
    if pd.isna(s):
        return 0

    # numeric input (already a count)
    if isinstance(s, (int, float)):
        try:
            if np.isnan(s):
                return 0
            return int(float(s))
        except Exception:
            return 0

    if not isinstance(s, str):
        # try to coerce to number
        try:
            return int(float(s))
        except Exception:
            return 0

    ss = s.strip().lower()
    if ss == "":
        return 0

    # split on common separators
    tokens = [t.strip() for t in re.split(r"[|,;/]+", ss) if t.strip() != ""]
    if not tokens:
        return 0

    # Conservative default: only count tokens that explicitly indicate a dedicated PSV lane.
    psv_explicit = {"designated", "exclusive"}

    count = 0
    for t in tokens:
        # numeric token -> add numeric value
        if re.match(r"^[0-9]+$", t):
            count += int(t)
            continue

        # strip common prefixes
        t_clean = re.sub(r'^(psv:|bus:)', '', t)

        # only accept explicit tokens that unambiguously mark a dedicated PSV lane
        if t_clean in psv_explicit:
            count += 1

    return max(0, int(count))

def _safe_min(a, b):
    try:
        a_missing = pd.isna(a)
        b_missing = pd.isna(b)
        if a_missing and b_missing:
            return np.nan

        a_val = 0 if a_missing else int(float(a))
        b_val = 0 if b_missing else int(float(b))
        return int(max(0, min(a_val, b_val)))
    except Exception:
        return np.nan

# 2) PSV-reserved lanes per direction
# Numeric counts first
psv_tot = pd.to_numeric(network_gdf.get("lanes:psv"), errors="coerce")
# ensure Series (preserve alignment with network_gdf)
psv_tot = pd.Series(psv_tot, index=network_gdf.index)

psv_f   = pd.to_numeric(network_gdf.get("lanes:psv:forward"), errors="coerce")
psv_f   = pd.Series(psv_f, index=network_gdf.index)

psv_b   = pd.to_numeric(network_gdf.get("lanes:psv:backward"), errors="coerce")
psv_b   = pd.Series(psv_b, index=network_gdf.index)

# Token strings (fallbacks)
tok_any = network_gdf.get("psv:lanes")
tok_any = pd.Series(tok_any, index=network_gdf.index)

tok_f   = network_gdf.get("psv:lanes:forward")
tok_f   = pd.Series(tok_f, index=network_gdf.index)

tok_b   = network_gdf.get("psv:lanes:backward")
tok_b   = pd.Series(tok_b, index=network_gdf.index)

# Some data uses bus:* instead of psv:*
tok_any_bus = network_gdf.get("bus:lanes")
tok_any_bus = pd.Series(tok_any_bus, index=network_gdf.index)

tok_f_bus   = network_gdf.get("bus:lanes:forward")
tok_f_bus   = pd.Series(tok_f_bus, index=network_gdf.index)

tok_b_bus   = network_gdf.get("bus:lanes:backward")
tok_b_bus   = pd.Series(tok_b_bus, index=network_gdf.index)

# Start with explicit directional numeric counts
psv_forward_dir  = psv_f.copy()
psv_backward_dir = psv_b.copy()

# If only total numeric count exists, apportion by directional lane share
mask_tot_only = psv_forward_dir.isna() & psv_backward_dir.isna() & psv_tot.notna()
if mask_tot_only.any():
    tot_psv = psv_tot.loc[mask_tot_only].astype(float)
    lf_share = lanes_forward_dir.loc[mask_tot_only].replace(0, np.nan)
    lb_share = lanes_backward_dir.loc[mask_tot_only].replace(0, np.nan)
    denom = (lf_share + lb_share)
    f_alloc = np.floor(tot_psv * (lf_share / denom)).fillna(0)
    b_alloc = (tot_psv - f_alloc).clip(lower=0)
    psv_forward_dir.loc[mask_tot_only]  = f_alloc
    psv_backward_dir.loc[mask_tot_only] = b_alloc

# If still NaN, parse token strings per direction
mask_need_tokens_f = psv_forward_dir.isna()
if mask_need_tokens_f.any():
    src = tok_f.where(tok_f.notna(), tok_any).where(lambda s: s.notna(), tok_any_bus)
    psv_forward_dir.loc[mask_need_tokens_f] = src.loc[mask_need_tokens_f].map(_count_psv_from_token_string)

mask_need_tokens_b = psv_backward_dir.isna()
if mask_need_tokens_b.any():
    src = tok_b.where(tok_b.notna(), tok_any).where(lambda s: s.notna(), tok_b_bus)
    psv_backward_dir.loc[mask_need_tokens_b] = src.loc[mask_need_tokens_b].map(_count_psv_from_token_string)

# Default zeros where still missing
psv_forward_dir  = psv_forward_dir.fillna(0).astype(int)
psv_backward_dir = psv_backward_dir.fillna(0).astype(int)

# Cap PSV counts by available lanes per direction
psv_forward_dir  = np.minimum(psv_forward_dir,  lanes_forward_dir).astype(int)
psv_backward_dir = np.minimum(psv_backward_dir, lanes_backward_dir).astype(int)

# 3) Final outputs per direction
network_gdf["lanes_psv_forward"] = psv_forward_dir.astype(int)
network_gdf["lanes_psv_backward"] = psv_backward_dir.astype(int)
network_gdf["lanes_general_forward"] = (lanes_forward_dir - psv_forward_dir).clip(lower=0).astype(int)
network_gdf["lanes_general_backward"] = (lanes_backward_dir - psv_backward_dir).clip(lower=0).astype(int)

# 4) Final outputs without directionality
# For network_gdf
network_gdf["lanes_psv"] = (network_gdf["lanes_psv_forward"] + network_gdf["lanes_psv_backward"]).astype(int)
network_gdf["lanes_general"] = (network_gdf["lanes_general_forward"] + network_gdf["lanes_general_backward"]).astype(int)

# Force tunnels and bridges to have 0 lanes (accept "yes" strings and boolean-like values)
s_tunnel = network_gdf["tunnel"] if "tunnel" in network_gdf.columns else pd.Series([pd.NA] * len(network_gdf), index=network_gdf.index)
s_bridge = network_gdf["bridge"] if "bridge" in network_gdf.columns else pd.Series([pd.NA] * len(network_gdf), index=network_gdf.index)

def bool_like_mask(s):
    # True or 1
    m = s.eq(True) | s.eq(1)
    # string-ish true values: "yes", "true", "1" (case-insensitive)
    m |= s.fillna("").astype(str).str.lower().isin(["yes", "true", "1"])
    return m.fillna(False)

mask_tunnel = bool_like_mask(s_tunnel)
mask_bridge = bool_like_mask(s_bridge)
mask_tb = mask_tunnel | mask_bridge

network_gdf.loc[mask_tb, ["lanes", "lanes_psv", "lanes_general"]] = 0
network_gdf.loc[mask_tb, ["lanes_psv_forward", "lanes_psv_backward",
                          "lanes_general_forward", "lanes_general_backward"]] = 0

# Ensure numeric maxspeed for existing data
network_gdf["maxspeed"] = pd.to_numeric(network_gdf["maxspeed"], errors="coerce")

# Ensure length exists (in meters) for weighted averaging
if "length" not in network_gdf.columns:
    network_gdf["length"] = network_gdf.geometry.length.astype(float)

# 1) Keep existing maxspeed as-is (already numeric)
# 2) Fill missing from user-provided dictionary
mask_missing = network_gdf["maxspeed"].isnull()
if mask_missing.any():
    mapped = network_gdf.loc[mask_missing, "highway"].map(speed_limits)
    mapped = pd.to_numeric(mapped, errors="coerce")
    network_gdf.loc[mask_missing, "maxspeed"] = mapped

# 3) Fallback: mode by highway type
# Compute mode of "maxspeed" for each "highway" type
valid_speed = network_gdf[network_gdf["maxspeed"].notnull()].copy()
valid_speed["maxspeed"] = pd.to_numeric(valid_speed["maxspeed"], errors="coerce")

mode_maxspeed = (
    valid_speed.groupby("highway")["maxspeed"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

# Fill missing "maxspeed" values using the mode for each "highway"
def fill_maxspeed(row):
    if pd.isnull(row["maxspeed"]):
        return mode_maxspeed.get(row["highway"], np.nan)
    return row["maxspeed"]

network_gdf["maxspeed"] = network_gdf.apply(fill_maxspeed, axis=1)
# convert to numeric (ints) where possible
network_gdf["maxspeed"] = pd.to_numeric(network_gdf["maxspeed"], errors="coerce").astype("Float64")

_link_map = {
    "motorway_link": "motorway",
    "trunk_link": "trunk",
    "primary_link": "primary",
    "secondary_link": "secondary",
    "tertiary_link": "tertiary"
}

network_gdf["highway_class"] = network_gdf["highway"].map(_link_map).fillna(network_gdf["highway"])

highway_priority = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "living_street",
    "pedestrian"
]

street_lines = neatnet.neatify(network_gdf, exclusion_mask = exclusion_mask.geometry,)

def assign_ids(gdf, col_name):
    gdf = gdf.reset_index(drop=True).copy()
    gdf[col_name] = gdf.index.astype(str)
    return gdf

network_gdf = assign_ids(network_gdf, 'orig_id')
street_lines = assign_ids(street_lines, 'street_id')

street_lines_new = street_lines.loc[street_lines['_status'] == 'new', ['street_id', '_status', 'geometry']].copy()
street_lines_new = gpd.GeoDataFrame(street_lines_new, crs=street_lines.crs)

probe_half_length = 50   # 50 meters to each side (total 100-meter probe)
probe_spacing     = 10   # probe every 10 meters along the line

# dissolve to one geometry per street_id
dissolved = street_lines_new.dissolve(by='street_id', as_index=False)

probe_recs = []
for _, row in dissolved.iterrows():
    sid  = row.street_id
    geom = row.geometry
    branches = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
    for seg in branches:
        seg_len = seg.length
        if seg_len <= probe_spacing:
            continue
        dists = np.arange(probe_spacing, seg_len, probe_spacing)

        # unit perpendicular from endpoints
        p0, p1 = Point(seg.coords[0]), Point(seg.coords[-1])
        dx, dy = p1.x - p0.x, p1.y - p0.y
        ux, uy = -dy, dx
        norm   = (ux**2 + uy**2)**0.5
        if norm == 0:
            continue
        ux, uy = ux/norm, uy/norm

        for d in dists:
            mid  = seg.interpolate(d)
            end1 = Point(mid.x + ux*probe_half_length, mid.y + uy*probe_half_length)
            end2 = Point(mid.x - ux*probe_half_length, mid.y - uy*probe_half_length)
            probe_recs.append({'street_id': sid, 'geometry': LineString([end2, end1])})

probes = gpd.GeoDataFrame(probe_recs, crs=street_lines_new.crs)
probes['probe_seq'] = probes.groupby('street_id').cumcount()
probes['probe_id']  = probes['street_id'] + '_' + probes['probe_seq'].astype(str)
probes = probes.drop(columns='probe_seq')

probes = probes.to_crs(network_gdf.crs)
sidx   = network_gdf.sindex

orig_ids_list = []
for g in probes.geometry:
    cand_idx = list(sidx.intersection(g.bounds))
    if not cand_idx:
        orig_ids_list.append([])
        continue
    cand = network_gdf.iloc[cand_idx]
    hits = cand[cand.intersects(g)]
    orig_ids_list.append(hits['orig_id'].tolist())

probes['orig_ids'] = orig_ids_list

exploded = (
    probes[['probe_id','street_id','geometry','orig_ids']]
    .explode('orig_ids')
    .dropna(subset=['orig_ids'])
    .rename(columns={'orig_ids':'orig_id'})
)

merged = exploded.merge(
    network_gdf.drop(columns='geometry'),
    on='orig_id',
    how='left'
)

# Defining mode or list custom function (for "name")
def mode_or_list(series):
    if series.empty:
        return np.nan
    mode_series = series.mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    else:
        return series.tolist()

# Defining mode or max custom function (for "maxspeed")
def mode_or_max(series):
    if series.nunique() == 1:
        return series.iloc[0]
    else:
        return series.mode().iloc[0] if not series.mode().empty else series.max()

# Defining list custom function (for "osmid")
def list_unique_flat(series):
    vals = []
    for v in series.dropna():
        if isinstance(v, (builtins.list, builtins.tuple, builtins.set)):
            vals.extend(v)
        else:
            vals.append(v)
    seen = set()
    out = []
    for item in vals:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

# Aggregating to the probes from the orig_id (using the custom functions, depending on the column)
# We're still on probe-level aggregation here
aggregated = merged.groupby(['probe_id', 'street_id', 'geometry']).agg({
    'name': mode_or_list,
    'osmid': list_unique_flat,
    'highway': mode_or_list,
    'maxspeed': mode_or_max,
    'lanes_psv': 'sum',
    'lanes_general': 'sum',
}).reset_index()

# Defining the mode or highway_priority hierarchy custom function (for "highway_class")
def mode_or_hierarchy(series):
    if series.empty:
        return np.nan
    mode_series = series.mode()
    if not mode_series.empty:
        return mode_series.iloc[0]
    else:
        return series.tolist()

# Defining mode or min custom function (for "lanes_psv", "lanes_general")
def mode_or_min(series):
    if series.empty:
        return None
    mode = series.mode()
    if not mode.empty:
        return mode[0]
    return series.min()

# Aggregating to the streets through the probes (with a "maxspeed" adjustment for user-provided limits)

# 1) aggregate with a named maxspeed from data
street_lines_new = aggregated.groupby('street_id').agg(
    name=('name', mode_or_list),
    osmid=('osmid', list_unique_flat),
    highway=('highway', mode_or_hierarchy),
    maxspeed_mode=('maxspeed', mode_or_max),
    lanes_psv=('lanes_psv', mode_or_min),
    lanes_general=('lanes_general', mode_or_min),
).reset_index()

# 2) override with user-provided limits, fallback to mode_or_max
#    priority: speed_limits[highway] -> maxspeed_mode
street_lines_new['maxspeed_user'] = street_lines_new['highway'].map(speed_limits)
street_lines_new['maxspeed'] = street_lines_new['maxspeed_user'].fillna(street_lines_new['maxspeed_mode'])

# clean up and ensure numeric
street_lines_new.drop(columns=['maxspeed_user', 'maxspeed_mode'], inplace=True)
street_lines_new['maxspeed'] = pd.to_numeric(street_lines_new['maxspeed'], errors='coerce').astype('Float64')

# columns produced by the probe pipeline
cols = ['name','osmid','highway','maxspeed','lanes_psv','lanes_general']

# attach new attributes next to the full network
aug = street_lines.merge(
    street_lines_new[['street_id'] + cols].add_suffix('_new').rename(columns={'street_id_new':'street_id'}),
    on='street_id',
    how='left'
)

# replace attributes only where _status == 'new'
mask = aug['_status'].eq('new')
for c in cols:
    aug[c] = np.where(mask, aug[f'{c}_new'], aug[c])

# clean up
aug = aug.drop(columns=[f'{c}_new' for c in cols])

# optional: ensure maxspeed dtype
aug['maxspeed'] = pd.to_numeric(aug['maxspeed'], errors='coerce').astype('Float64')

# result: full network with “new” rows updated
street_lines = aug

# Get "length" back in the street_lines GeoDataFrame
street_lines['length'] = street_lines.geometry.length.astype(float)

# Static map output
fig, ax = plt.subplots(figsize=(10, 10))

# --- Study area ---
study_area_gdf.plot(
    ax=ax,
    facecolor="none",  # polygon transparent fill
    linewidth=1,
    edgecolor="#000000",
    alpha=1
)

# --- Street lines by status ---
# colors: original (blue), new (green), changed (greenish-blue/teal)
status_colors = {
    "original": "#00AEFF",
    "changed": "#3700FF",
    "new": "#EA00FF"
}

for status, color in status_colors.items():
    subset = street_lines[street_lines["_status"].str.lower() == status]
    if not subset.empty:
        subset.plot(
            ax=ax,
            edgecolor=color,
            linewidth=1,
            label=status.capitalize(),
            zorder=5
        )


# --- Basemap (kept underneath everything) ---
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=local_CRS, zorder=0)

# --- Legend ---
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=status_colors["original"], lw=2, label="Original"),
    Line2D([0], [0], color=status_colors["changed"], lw=2, label="Changed"),
    Line2D([0], [0], color=status_colors["new"], lw=2, label="New"),
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=True,
    framealpha=0.9
)

ax.set_axis_off()
ax.set_aspect('equal')

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = study_area_gdf.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [2000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

# Calculate descriptive statistics for the base street centerlines network

# Total number of segments
total_segments = len(street_lines)
# Number of segments by status
status_counts = street_lines["_status"].value_counts()
original_count = status_counts.get("original", 0)
changed_count = status_counts.get("changed", 0)
new_count = status_counts.get("new", 0)

# Percentage by status
original_pct = 100 * original_count / total_segments if total_segments else 0
changed_pct = 100 * changed_count / total_segments if total_segments else 0
new_pct = 100 * new_count / total_segments if total_segments else 0

# Calculate length (in meters) for each segment
if "length" not in street_lines.columns:
    street_lines["length"] = street_lines.geometry.length

# Total network length (km)
total_length_km = street_lines["length"].sum() / 1000
# Length by status (km)
original_length_km = street_lines.loc[street_lines["_status"] == "original", "length"].sum() / 1000
changed_length_km = street_lines.loc[street_lines["_status"] == "changed", "length"].sum() / 1000
new_length_km = street_lines.loc[street_lines["_status"] == "new", "length"].sum() / 1000

# Percentage of length by status
original_length_pct = 100 * original_length_km / total_length_km if total_length_km else 0
changed_length_pct = 100 * changed_length_km / total_length_km if total_length_km else 0
new_length_pct = 100 * new_length_km / total_length_km if total_length_km else 0

# Average segment length (m)
avg_length_m = street_lines["length"].mean()

# Display results

stats_df = pd.DataFrame({
    "Metric": [
        "Number of segments",
        "Original",
        "Changed",
        "New",
        "Network length (km)",
        "Original",
        "Changed",
        "New",
        "Average length of all segments (m)"
    ],
    "Value": [
        total_segments,
        original_count,
        changed_count,
        new_count,
        round(total_length_km, 2),
        round(original_length_km, 2),
        round(changed_length_km, 2),
        round(new_length_km, 2),
        round(avg_length_m, 2)
    ],
    "Percentage (%)": [
        100,
        round(original_pct, 2),
        round(changed_pct, 2),
        round(new_pct, 2),
        100,
        round(original_length_pct, 2),
        round(changed_length_pct, 2),
        round(new_length_pct, 2),
        ""
    ]
})

display(stats_df)

# Defining general lane base capacity per "highway" type,between 600 and 1600 p/h/lane
general_lane_base_capacity = {
    "motorway": 1600,
    "motorway_link": 1500,
    "trunk": 1400,
    "trunk_link": 1300,
    "primary": 1200,
    "primary_link": 1100,
    "secondary": 1000,
    "secondary_link": 900,
    "tertiary": 800,
    "tertiary_link": 700,
    "unclassified": 600,
    "residential": 600,
    "living_street": 600,
    "pedestrian": 600   
}

# Defining PSV lane base capacity per "highway" type, between 1000 and 2800 p/h/lane
psv_lane_base_capacity = {
    "motorway": 2800,
    "motorway_link": 2620,
    "trunk": 2440,
    "trunk_link": 2260,
    "primary": 2080,
    "primary_link": 1900,
    "secondary": 1720,
    "secondary_link": 1540,
    "tertiary": 1360,
    "tertiary_link": 1180,
    "unclassified": 1000,
    "residential": 1000,
    "living_street": 1000,
    "pedestrian": 1000  # If a pedestrian street has PSV lanes, it is seen as low-level
}

# Calculating potential capacities
street_lines["pot_capacity"] = street_lines["lanes_general"] * street_lines["highway"].map(general_lane_base_capacity).fillna(0) +  \
                               street_lines["lanes_psv"] * street_lines["highway"].map(psv_lane_base_capacity).fillna(0)

def endpoints(geom):
    if isinstance(geom, LineString):
        c = geom.coords
        return [((c[0][0], c[0][1]), (c[-1][0], c[-1][1]))]
    elif isinstance(geom, MultiLineString):
        pairs = []
        for part in geom.geoms:
            c = part.coords
            pairs.append(((c[0][0], c[0][1]), (c[-1][0], c[-1][1])))
        return pairs
    else:
        return []

street_lines_MultiGraph = nx.Graph()
for sid, geom, L in zip(street_lines["street_id"], street_lines.geometry, street_lines["length"]):
    for a, b in endpoints(geom):   # iterate through all pairs
        street_lines_MultiGraph.add_edge(a, b, street_id=sid, length=float(L))

edge_bc = nx.edge_betweenness_centrality(street_lines_MultiGraph, 
                                         weight = "length",
                                         normalized = "True"
                                         )

id_to_bc = {}

for (u, v, data) in street_lines_MultiGraph.edges(data=True):
    sid = data["street_id"]
    # For an undirected Graph, edge key is (u,v) or (v,u) — use either
    id_to_bc[sid] = edge_bc.get((u, v), edge_bc.get((v, u)))

street_lines["bet_centrality"] = street_lines["street_id"].map(id_to_bc)

street_lines["bet_centrality"] = street_lines["bet_centrality"].fillna(0)

street_lines["highway_level"] = street_lines["highway"].map({
    "pedestrian": 1,
    "living_street": 2,
    "residential": 3,
    "unclassified": 3,
    "tertiary": 4,
    "tertiary_link": 4,
    "secondary": 5,
    "secondary_link": 5,
    "primary": 6,
    "primary_link": 6,
    "trunk": 7,
    "trunk_link": 7,
    "motorway": 8,
    "motorway_link": 8
})

# Handling segments with no highway
street_lines["failed_data"] = street_lines["highway"].isna()

P99_bet_centrality = street_lines["bet_centrality"].quantile(0.99)

street_lines["winsorized_bet_centrality"] = street_lines["bet_centrality"].clip(upper=P99_bet_centrality)

# Plot 1: Potential capacity
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(street_lines['pot_capacity'].dropna(), bins=30, color="#FF3333", edgecolor='white')
ax.set_xlabel('(bins = 30)', labelpad=10, size=14)
plt.tight_layout()
plt.show()

# Plot 2: Winsorized betweenness centrality
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(street_lines['winsorized_bet_centrality'].dropna(), bins=30, color="#FF3333", edgecolor='white')
ax.set_xlabel('(bins = 30)', labelpad=10, size=14)
plt.tight_layout()
plt.show()

# Plot 3: Highway level (discrete bins)
hl = street_lines['highway_level'].dropna()
if not hl.empty:
    bins_hl = np.arange(hl.min() - 0.5, hl.max() + 1.5, 1)
else:
    bins_hl = 8  # fallback
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(hl, bins=bins_hl, color="#FF3333", edgecolor='white')
ax.set_xlabel('(bins = 8)', labelpad=10, size=14)
plt.tight_layout()
plt.show()

# Plot 4: Speed limits
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(street_lines['maxspeed'].dropna(), bins=12, color="#FF3333", edgecolor='white')
ax.set_xlabel('(bins = 12)', labelpad=10, size=14)
plt.tight_layout()
plt.show()

# Transposed descriptive statistics: stats as rows, attributes as columns
attributes = ["pot_capacity", "winsorized_bet_centrality", "highway_level", "maxspeed"]
stats_order = ["Count", "Mode", "Mean", "Std. Dev.", "Median", "Min", "Max", "P25", "P75", "Skewness", "Kurtosis"]

data = {}
for col in attributes:
    series = street_lines[col].dropna()
    if series.empty:
        vals = [
            0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        ]
    else:
        # mode: pick first mode if exists
        mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
        vals = [
            series.count(),
            mode_val,
            series.mean(),
            series.std(),
            series.median(),
            series.min(),
            series.max(),
            series.quantile(0.25),
            series.quantile(0.75),
            series.skew(),
            series.kurtosis()
        ]
    data[col] = vals

desc_stats_transposed = pd.DataFrame(data, index=stats_order)
display(desc_stats_transposed)

# Z-score normalization of potential capacity
street_lines["z_pot_capacity"] = (
    (street_lines["pot_capacity"] - street_lines["pot_capacity"].mean()) /
    street_lines["pot_capacity"].std()
)

# Z-score normalization of winsorized edge betweenness centrality
street_lines["z_bet_centrality"] = (
    (street_lines["winsorized_bet_centrality"] - street_lines["winsorized_bet_centrality"].mean()) /
    street_lines["winsorized_bet_centrality"].std()
)

# Z-score normalization of highway level
street_lines["z_highway_level"] = (
    (street_lines["highway_level"] - street_lines["highway_level"].mean()) /
    street_lines["highway_level"].std()
)

# Z-score normalization of speed limits
street_lines["z_maxspeed"] = (
    (street_lines["maxspeed"] - street_lines["maxspeed"].mean()) /
    street_lines["maxspeed"].std()
)

# Calculating Pearson correlation between the four z-score normalized metrics
corr_cap_cen = street_lines[["z_pot_capacity", "z_bet_centrality"]].corr(method='pearson').iloc[0, 1]
corr_cap_lev = street_lines[["z_pot_capacity", "z_highway_level"]].corr(method='pearson').iloc[0, 1]
corr_cap_spd = street_lines[["z_pot_capacity", "z_maxspeed"]].corr(method='pearson').iloc[0, 1]
corr_cen_lev = street_lines[["z_bet_centrality", "z_highway_level"]].corr(method='pearson').iloc[0, 1]
corr_cen_spd = street_lines[["z_bet_centrality", "z_maxspeed"]].corr(method='pearson').iloc[0, 1]
corr_lev_spd = street_lines[["z_highway_level", "z_maxspeed"]].corr(method='pearson').iloc[0, 1]

def _plot_with_trend(ax, x, y, xlabel, ylabel, title):
    mask = x.notna() & y.notna()
    ax.scatter(x[mask], y[mask], alpha=0.5, color="#FF3333")
    # add linear trendline if enough points
    if mask.sum() > 1:
        xp = x[mask].values
        yp = y[mask].values
        coeff = np.polyfit(xp, yp, 1)
        xs_sorted = np.sort(xp)
        ys_fit = np.polyval(coeff, xs_sorted)
        ax.plot(xs_sorted, ys_fit, color="#640000", linestyle=':', linewidth=2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

# Plot 1
fig, ax = plt.subplots(figsize=(6, 6))
_plot_with_trend(
    ax,
    street_lines["z_pot_capacity"],
    street_lines["z_bet_centrality"],
    "Potential Capacity (z-score)",
    "Winsorized Betweenness Centrality (z-score)",
    f"Pearson Correlation: {corr_cap_cen:.4f}"
)
fig.tight_layout()
plt.show()

# Plot 2
fig, ax = plt.subplots(figsize=(6, 6))
_plot_with_trend(
    ax,
    street_lines["z_pot_capacity"],
    street_lines["z_highway_level"],
    "Potential Capacity (z-score)",
    "Highway Level (z-score)",
    f"Pearson Correlation: {corr_cap_lev:.4f}"
)
fig.tight_layout()
plt.show()

# Plot 3
fig, ax = plt.subplots(figsize=(6, 6))
_plot_with_trend(
    ax,
    street_lines["z_pot_capacity"],
    street_lines["z_maxspeed"],
    "Potential Capacity (z-score)",
    "Speed Limits (z-score)",
    f"Pearson Correlation: {corr_cap_spd:.4f}"
)
fig.tight_layout()
plt.show()

# Plot 4
fig, ax = plt.subplots(figsize=(6, 6))
_plot_with_trend(
    ax,
    street_lines["z_bet_centrality"],
    street_lines["z_highway_level"],
    "Winsorized Betweenness Centrality (z-score)",
    "Highway Level (z-score)",
    f"Pearson Correlation: {corr_cen_lev:.4f}"
)
fig.tight_layout()
plt.show()

# Plot 5
fig, ax = plt.subplots(figsize=(6, 6))
_plot_with_trend(
    ax,
    street_lines["z_bet_centrality"],
    street_lines["z_maxspeed"],
    "Winsorized Betweenness Centrality (z-score)",
    "Speed Limits (z-score)",
    f"Pearson Correlation: {corr_cen_spd:.4f}"
)
fig.tight_layout()
plt.show()

# Plot 6
fig, ax = plt.subplots(figsize=(6, 6))
_plot_with_trend(
    ax,
    street_lines["z_highway_level"],
    street_lines["z_maxspeed"],
    "Highway Level (z-score)",
    "Speed Limits (z-score)",
    f"Pearson Correlation: {corr_lev_spd:.4f}"
)
fig.tight_layout()
plt.show()

cols = ["z_pot_capacity", "z_bet_centrality", "z_highway_level", "z_maxspeed"]
corr_df = street_lines[cols].corr().round(4)
corr_df

# Maps of z-score normalized LINK metrics

cmap = LinearSegmentedColormap.from_list("mygrad", link_palette_grad, N=256)

# reproject once to Web Mercator for basemap compatibility
street_lines_3857 = street_lines.to_crs(epsg=3857)

# prepare a mask for failed data (safe handling of missing/NaN)
failed_mask = street_lines_3857.get('failed_data', False).fillna(False).astype(bool)

# Plot 1: Potential Capacity of Streets (z-score)
fig, ax = plt.subplots(figsize=(10, 10))
# plot valid data with colormap, then plot failed segments in grey on top
street_lines_3857[~failed_mask].plot(column='z_pot_capacity', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color='#808080', linewidth=1)  # grey for failed data
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()


# Plot 2: Betweenness Centrality of Streets (z-score)
fig, ax = plt.subplots(figsize=(10, 10))
street_lines_3857.plot(column='z_bet_centrality', ax=ax, legend=False, cmap=cmap, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()


# Plot 3: Highway Level of Streets (z-score)
fig, ax = plt.subplots(figsize=(10, 10))
# plot valid data with colormap, then failed in grey
street_lines_3857[~failed_mask].plot(column='z_highway_level', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color='#808080', linewidth=1)  # grey for failed data
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

# Plot 4: Speed Limits of Streets (z-score)
fig, ax = plt.subplots(figsize=(10, 10))
# plot valid data with colormap, then failed in grey
street_lines_3857[~failed_mask].plot(column='z_maxspeed', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color='#808080', linewidth=1)  # grey for failed data
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

pca_data = street_lines[['z_pot_capacity', 'z_bet_centrality', 'z_highway_level', 'z_maxspeed']].dropna()
pca = PCA()
pca.fit(pca_data)
explained_variance = pca.explained_variance_ratio_
cum_variance = np.cumsum(explained_variance)

fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

indices = np.arange(1, len(explained_variance) + 1)
width = 0.6

# Bars: explained variance ratio on primary y-axis
bars = ax1.bar(indices, explained_variance, color="#FF8787", width=width,
    label='Explained Variance Ratio', alpha=0.9, zorder=1)

# Line with markers: cumulative explained variance on secondary y-axis (draw after, higher zorder -> in front of bars)
line = ax2.plot(indices, cum_variance, marker='o', color="#D81E1E",
     linewidth=2.5, label='Cumulative Explained Variance', zorder=5)[0]

# Ensure ax2 is above ax1 so the line appears in front; make ax2 background transparent
ax1.set_zorder(1)
ax2.set_zorder(2)
ax2.patch.set_visible(False)

# X axis formatting
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xticks(indices)
ax1.set_xticklabels([str(int(x)) for x in indices])
ax1.set_xlabel('Principal Components')

# Y axis labels and limits
ax1.set_ylim(0, 1.0)
ax2.set_ylim(0, 1.0)

# Combined legend placed on top of the plot
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

# Make room on top for the external legend
fig.subplots_adjust(top=0.85)

plt.tight_layout()
plt.show()

# create table of PCA eigenvalues and variance ratios
idx = [f'PC{i+1}' for i in range(pca.n_components_)]
df_pca_summary = pd.DataFrame({
    "Eigenvalues": pca.explained_variance_,
    "Explained Variance Ratio": pca.explained_variance_ratio_,
    "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_)
}, index=idx).round(4)

df_pca_summary

# Show loading scores in a table (rows = features, columns = principal components)
n_disp = min(4, pca.n_components_)
cols = [f'PC{i+1}' for i in range(n_disp)]
loadings_df = pd.DataFrame(pca.components_[:n_disp].T, index=pca_data.columns, columns=cols)
loadings_df = loadings_df.round(4)
loadings_df

# plain component loadings for PC1 (use this for simple coefficient-based weights)
loading_scores_pc1 = pd.Series(pca.components_[0], index=pca_data.columns)

# Calculating final weight of each metric in the first principal component
total_loading_pc1 = loading_scores_pc1.abs().sum()
weight_pc1 = loading_scores_pc1.abs() / total_loading_pc1
print("Final Weights of Each Metric in the First Principal Component:")
print(weight_pc1)

# Calculating final link function with the assessed weights
street_lines["link_function"] = (
    street_lines["z_pot_capacity"] * weight_pc1["z_pot_capacity"] +
    street_lines["z_bet_centrality"] * weight_pc1["z_bet_centrality"] +
    street_lines["z_highway_level"] * weight_pc1["z_highway_level"] +
    street_lines["z_maxspeed"] * weight_pc1["z_maxspeed"]
)

# Normalizing values to a 0-1 scale
min_link = street_lines["link_function"].min()
max_link = street_lines["link_function"].max()
street_lines["LINK"] = (street_lines["link_function"] - min_link) / (max_link - min_link)

# Histogram of LINK
plt.figure(figsize=(18, 5))
plt.hist(street_lines['LINK'], bins=100, color="#FF3333", edgecolor='white')
plt.xlabel('bins = 100', labelpad=10, size=14)
plt.show()

# Compute LINK descriptive stats without using stats_order; exclude min/max and include P90
s = series.dropna() if "series" in globals() else street_lines["LINK"].dropna()

measures = ["Count", "Mode", "Mean", "Std. Dev.", "Median", "P25", "P75", "P90", "Skewness", "Kurtosis"]

if s.empty:
    vals2 = [0] + [np.nan] * (len(measures) - 1)
else:
    mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
    # Count rounded to units (no decimals)
    count_val = int(round(s.count(), 0))
    vals2 = [
        count_val,
        mode_val,
        s.mean(),
        s.std(),
        s.median(),
        s.quantile(0.25),
        s.quantile(0.75),
        s.quantile(0.90),
        s.skew(),
        s.kurtosis(),
    ]

# Helper to convert numpy/pandas scalar types to native Python types
def _py_val(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_py_val(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

# Convert to Python-native types for safe display/serialization
values = [_py_val(v) for v in vals2]

df_link_stats = pd.DataFrame({
    "Measure": measures,
    "Value": values
})

df_link_stats

# Setting the colors
cmap = LinearSegmentedColormap.from_list("linkgrad", link_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of the final link function (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='LINK', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['LINK'].min(), street_lines_3857['LINK'].max())
sm.set_array(street_lines_3857['LINK'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

point_poi = {
    "amenity": [
        "bar","biergarten","cafe","fast_food","food_court","pub","restaurant",
        "college","dancing_school","driving_school","first_aid_school","kindergarten",
        "language_school","library","surf_school","toy_library","research_institute",
        "training","music_school","school","traffic_park","university",
        "bicycle_repair_station","bicycle_wash","car_wash","vehicle_inspection","fuel","taxi",
        "bank","bureau_de_change","money_transfer","payment_centre",
        "clinic","dentist","doctors","hospital","nursing_home","pharmacy","social_facility","veterinary",
        "arts_centre","brothel","casino","cinema","community_centre","conference_centre",
        "events_venue","exhibition_centre","gambling","love_hotel",
        "music_venue","nightclub","planetarium","social_centre","stage","stripclub",
        "studio","theatre","courthouse","fire_station","police","post_office","townhall",
        "animal_shelter","crematorium","dive_centre","funeral_hall","internet_cafe",
        "marketplace","monastery","mortuary","place_of_worship","public_bath"
    ],
    "shop": [
        "supermarket","convenience","bakery","butcher","greengrocer",
        "clothes","shoes","department_store","mall",
        "electronics","furniture","hardware",
        "pharmacy","optician",
        "hairdresser","beauty","laundry","dry_cleaning",
        "travel_agency","car_repair","car_rental","bicycle",
        "bookstore","stationery","gift","florist"
    ],
    "leisure": [
        "park","playground","sports_centre","pitch","stadium",
        "fitness_centre","swimming_pool","golf_course",
        "dance","ice_rink","marina","water_park","sauna"
    ],
    "tourism": [
        "aquarium","artwork","attraction",
        "camp_pitch","camp_site","caravan_site","chalet",
        "gallery","guest_house","hostel","hotel","information",
        "motel","museum","picnic_site","theme_park",
        "viewpoint","zoo"
    ],
    "public_transport": [
        "platform",
        "station"
    ],
    "highway": [
        "bus_stop"
    ],
    "healthcare": True,
}

# Retrieve POIs as points
poi_point_tags = {}
for key, val in point_poi.items():
    if val is True:
        poi_point_tags[key] = True
    else:
        poi_point_tags[key] = val

poi_points = ox.features_from_place(study_area, tags=poi_point_tags, which_result=None)

# Keep only point geometries
poi_points = poi_points[poi_points.geometry.type.isin(['Point', 'MultiPoint'])].copy()

# Assigning POI categories to each OSM tag
poi_category_map = {
    "amenity": {
        "bar": "food_and_drinks",
        "biergarten": "food_and_drinks",
        "cafe": "food_and_drinks",
        "fast_food": "food_and_drinks",
        "food_court": "food_and_drinks",
        "pub": "food_and_drinks",
        "restaurant": "food_and_drinks",
        "college": "education_and_knowledge",
        "dancing_school": "education_and_knowledge",
        "driving_school": "education_and_knowledge",
        "first_aid_school": "education_and_knowledge",
        "kindergarten": "education_and_knowledge",
        "language_school": "education_and_knowledge",
        "library": "education_and_knowledge",
        "surf_school": "education_and_knowledge",
        "toy_library": "education_and_knowledge",
        "research_institute": "education_and_knowledge",
        "training": "education_and_knowledge",
        "music_school": "education_and_knowledge",
        "school": "education_and_knowledge",
        "traffic_park": "education_and_knowledge",
        "university": "education_and_knowledge",
        "bicycle_repair_station": "services",
        "bicycle_wash": "services",
        "car_wash": "services",
        "vehicle_inspection": "services",
        "fuel": "transportation",
        "taxi": "transportation",
        "bank": "services",
        "bureau_de_change": "services",
        "money_transfer": "services",
        "payment_centre": "services",
        "clinic": "health_and_care",
        "dentist": "health_and_care",
        "doctors": "health_and_care",
        "hospital": "health_and_care",
        "nursing_home": "health_and_care",
        "pharmacy": "health_and_care",
        "social_facility": "health_and_care",
        "veterinary": "health_and_care",
        "arts_centre": "entertainment",
        "brothel": "services",
        "casino": "entertainment",
        "cinema": "entertainment",
        "community_centre": "government_and_public_services",
        "conference_centre": "government_and_public_services",
        "events_venue": "entertainment",
        "exhibition_centre": "entertainment",
        "gambling": "entertainment",
        "love_hotel": "services",
        "music_venue": "entertainment",
        "nightclub": "entertainment",
        "planetarium": "entertainment",
        "social_centre": "government_and_public_services",
        "stage": "entertainment",
        "stripclub": "entertainment",
        "studio": "entertainment",
        "theatre": "entertainment",
        "courthouse": "government_and_public_services",
        "fire_station": "government_and_public_services",
        "police": "government_and_public_services",
        "post_office": "government_and_public_services",
        "townhall": "government_and_public_services",
        "animal_shelter": "health_and_care",
        "crematorium": "other",
        "dive_centre": "entertainment",
        "funeral_hall": "other",
        "internet_cafe": "food_and_drinks",
        "marketplace": "shopping",
        "monastery": "other",
        "mortuary": "other",
        "place_of_worship": "other",
        "public_bath": "health_and_care"
    },
    "shop": {
        "supermarket": "shopping",
        "convenience": "shopping",
        "bakery": "food_and_drinks",
        "butcher": "food_and_drinks",
        "greengrocer": "food_and_drinks",
        "clothes": "shopping",
        "shoes": "shopping",
        "department_store": "shopping",
        "mall": "shopping",
        "electronics": "shopping",
        "furniture": "shopping",
        "hardware": "shopping",
        "pharmacy": "health_and_care",
        "optician": "health_and_care",
        "hairdresser": "services",
        "beauty": "services",
        "laundry": "services",
        "dry_cleaning": "services",
        "travel_agency": "services",
        "car_repair": "services",
        "car_rental": "transportation",
        "bicycle": "transportation",
        "bookstore": "shopping",
        "stationery": "shopping",
        "gift": "shopping",
        "florist": "shopping"
    },
    "leisure": {
        "park": "entertainment",
        "playground": "entertainment",
        "sports_centre": "entertainment",
        "pitch": "entertainment",
        "stadium": "entertainment",
        "fitness_centre": "entertainment",
        "swimming_pool": "entertainment",
        "golf_course": "entertainment",
        "dance": "entertainment",
        "ice_rink": "entertainment",
        "marina": "transportation",
        "water_park": "entertainment",
        "sauna": "health_and_care"
    },
    "tourism": {
        "aquarium": "entertainment",
        "artwork": "entertainment",
        "attraction": "entertainment",
        "camp_pitch": "entertainment",
        "camp_site": "entertainment",
        "caravan_site": "entertainment",
        "chalet": "entertainment",
        "gallery": "entertainment",
        "guest_house": "services",
        "hostel": "services",
        "hotel": "services",
        "information": "government_and_public_services",
        "motel": "services",
        "museum": "entertainment",
        "picnic_site": "entertainment",
        "theme_park": "entertainment",
        "viewpoint": "entertainment",
        "zoo": "entertainment"
    },
    "public_transport": {
        "platform": "transportation",
        "station": "transportation"
    },
    "highway": {
        "bus_stop": "transportation"
    },
    "healthcare": "health_and_care"
}

# Assign these categories to each POI in "poi_points" GeoDataFrame
def assign_poi_category(row):
    for key in poi_category_map.keys():
        if key in row and pd.notna(row[key]):
            val = row[key]
            if key == "healthcare" and val is True:
                return poi_category_map[key]
            elif val in poi_category_map[key]:
                return poi_category_map[key][val]
    return "other"

poi_points['poi_category'] = poi_points.apply(assign_poi_category, axis=1)

# Define buffer sizes for each highway type
buffer_sizes = {
    "motorway": 60,
    "trunk": 50,
    "primary": 40,
    "secondary": 35,
    "tertiary": 30,
    "residential": 25,
    "unclassified": 25,
    "living_street": 20,
    "pedestrian": 15
}

def get_buffer_size(row):
    base = buffer_sizes.get(str(row["highway"]).lower(), 30)  # default to 30 if not found
    if str(row["_status"]).lower() == "new":
        return base * 1.15
    return base

# Compute buffer size for each segment
street_lines["buffer_size"] = street_lines.apply(get_buffer_size, axis=1)

local_EPSG = CRS.from_user_input(local_CRS).to_epsg()

if street_lines.crs is None or street_lines.crs.to_epsg() != local_EPSG:
    street_lines_buffered = street_lines.to_crs(local_CRS).copy()
else:
    street_lines_buffered = street_lines.copy()

street_lines_buffered["geometry"] = street_lines_buffered.geometry.buffer(street_lines_buffered["buffer_size"])

# Project POIs to match the CRS of street_lines_buffered
poi_points_proj = poi_points.to_crs(street_lines_buffered.crs).reset_index(drop=True)

# Define allowed POI tags for motorway/trunk
allowed_tags = {"fuel", "bus_stop", "platform"}

# Helper function to check if a POI row matches allowed tags
def is_allowed_poi(row):
    # Check for 'fuel' in amenity, 'bus_stop' in highway, 'platform' in public_transport
    if row.get("amenity") == "fuel":
        return True
    if row.get("highway") == "bus_stop":
        return True
    if row.get("public_transport") == "platform":
        return True
    return False

# Mark allowed POIs in points and polygons
poi_points_proj["is_allowed"] = poi_points_proj.apply(is_allowed_poi, axis=1)

# Create a unique POI ID for each row (using index as string)
poi_points_proj = poi_points_proj.copy()
poi_points_proj["poi_id"] = poi_points_proj.index.astype(str)

# Prepare street buffer gdf while avoiding column name collision (rename highway)
street_right = street_lines_buffered[["street_id", "geometry", "highway"]].rename(columns={"highway": "street_highway"})

# Spatial join: which POIs intersect each street buffer
joined = gpd.sjoin(
    poi_points_proj,
    street_right,
    how="inner",
    predicate="intersects"
)

# For each row, keep only allowed POIs for motorway/trunk, else keep all
def filter_poi(row):
    # use renamed column from the right-hand GeoDataFrame
    if row.get("street_highway") in ["motorway", "trunk"]:
        return row.get("is_allowed", False)
    return True

joined = joined[joined.apply(filter_poi, axis=1)]

# Group by street_id and aggregate POI IDs as a list
poi_ids_by_street = joined.groupby("street_id")["poi_id"].agg(list)

# Assign to street_lines as a new column
street_lines["poi_ids"] = street_lines["street_id"].map(poi_ids_by_street).apply(lambda x: x if isinstance(x, list) else [])

# Count POIs per street segment and normalize per 100 meters
def count_pois_per_100m(row):
    poi_count = len(row["poi_ids"]) if isinstance(row["poi_ids"], list) else 0
    length_m = row["length"] if row["length"] else 1  # avoid division by zero
    count = poi_count / (length_m / 100)
    hwy = str(row.get("highway", "")).lower()
    return count

street_lines["pois_per_100m"] = street_lines.apply(count_pois_per_100m, axis=1)

street_lines["poi_density_transformed"] = np.arcsinh(street_lines["pois_per_100m"])

# 0-MAX normalization of transformed POI density
street_lines["poi_density_normalized"] = street_lines["poi_density_transformed"] / street_lines["poi_density_transformed"].max()

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of the final POI density (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='poi_density_normalized', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['poi_density_normalized'].min(), street_lines_3857['poi_density_normalized'].max())
sm.set_array(street_lines_3857['poi_density_normalized'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

# Calculating Shannon Entropy Index of POI Categories per street segment

def shannon_entropy(category_list):
    if not category_list:
        return 0.0
    counts = pd.Series(category_list).value_counts()
    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log2(proportions))
    k = len(counts)
    if k <= 1:
        return 0.0
    max_entropy = np.log2(k)
    return float(entropy)

# Mapping poi_id to  assigned custom category taxonomy
poi_cat_map = dict(
    zip(
        poi_points_proj["poi_id"].astype(str),
        poi_points_proj["poi_category"]
    )
)

def get_poi_categories(ids):
    if not ids:
        return []
    out = []
    for i in ids:
        v = poi_cat_map.get(str(i), None)
        if pd.notna(v):
            out.append(v)
    return out

street_lines["poi_categories"] = street_lines["poi_ids"].apply(get_poi_categories)
street_lines["poi_shannon_entropy"] = street_lines["poi_categories"].apply(shannon_entropy)
# 0-1 normalization of POI Shannon Entropy
max_entropy_value = street_lines["poi_shannon_entropy"].max()
street_lines["poi_shannon_entropy"] = street_lines["poi_shannon_entropy"] / max_entropy_value

# Histogram of POI Shannon Entropy
plt.figure(figsize=(18, 5))
plt.hist(street_lines['poi_shannon_entropy'], bins=30, color="#33FF33", edgecolor='white')
plt.xlabel('bins = 30', labelpad=10, size=14)
plt.show()

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of the final POI diversity (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='poi_shannon_entropy', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['poi_shannon_entropy'].min(), street_lines_3857['poi_shannon_entropy'].max())
sm.set_array(street_lines_3857['poi_shannon_entropy'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

# Green polygons
green_tags = {
    "landuse": [
        "forest", "vineyard", "plant_nursery", "orchard",
        "greenfield", "recreation_ground", "allotments",
        "meadow", "grass", "farmland"
    ],
    "leisure": [
        "park", "garden", "dog_park", "nature_reserve"
    ],
    "amenity": ["park"]
}

green_areas_raw = ox.features_from_place(study_area, tags=green_tags)
green_areas = (
    green_areas_raw
    .loc[green_areas_raw.geometry.type.isin(["Polygon", "MultiPolygon"])]
    .to_crs(local_CRS)
)

# Tree points
tree_tags = {"natural": "tree"}
trees_raw = ox.features_from_place(study_area, tags=tree_tags)
trees = (
    trees_raw
    .loc[trees_raw.geometry.type == "Point"]
    .to_crs(local_CRS)
)

def compute_green_index(
    streets: gpd.GeoDataFrame,
    green_areas: gpd.GeoDataFrame,
    trees: gpd.GeoDataFrame,
    crs: str = None,
    D: float = 100.0,
    buffer_distance: float = 120.0,
    area_scale_m2: float = 10_000.0,  # 1 ha
) -> gpd.GeoDataFrame:
    """
    Approximate greenR-like green index for street segments.

    Parameters
    ----------
    streets : GeoDataFrame
        Street segments (LineString / MultiLineString).
    green_areas : GeoDataFrame
        Green polygons (parks, forests, etc.).
    trees : GeoDataFrame
        Tree points.
    crs : str, optional
        Target projected CRS (e.g. 'epsg:3763'). If provided, all layers
        are reprojected to this CRS.
    D : float, optional
        Distance-decay parameter in meters. Larger D = slower decay.
    buffer_distance : float, optional
        Maximum distance (m) to consider green features as influencing a segment.
    area_scale_m2 : float, optional
        Divisor to scale polygon areas (e.g. 10_000 m² = 1 ha).

    Returns
    -------
    GeoDataFrame
        `streets` with added columns:
        - 'green_raw_poly'
        - 'green_raw_tree'
        - 'green_raw'
        - 'green_index'  # normalized to [0, 1] after log1p transform
    """

    streets = streets.copy()

    # Reproject to common CRS if requested
    if crs is not None:
        streets = streets.to_crs(crs)
        green_areas = green_areas.to_crs(crs)
        trees = trees.to_crs(crs)

    # Drop empty geometries
    streets = streets[streets.geometry.notnull() & ~streets.geometry.is_empty].copy()
    green_areas = green_areas[green_areas.geometry.notnull() & ~green_areas.geometry.is_empty].copy()
    trees = trees[trees.geometry.notnull() & ~trees.geometry.is_empty].copy()

    # Midpoints as representative locations
    streets["__midpoint"] = streets.geometry.interpolate(0.5, normalized=True)
    midpoints = streets["__midpoint"].values

    # Trees index
    tree_geoms = list(trees.geometry.values)
    tree_index = STRtree(tree_geoms) if len(tree_geoms) > 0 else None

    # Polygon index (use full polygons)
    poly_geoms = list(green_areas.geometry.values)
    if len(poly_geoms) > 0:
        poly_index = STRtree(poly_geoms)
        poly_areas = np.array([geom.area for geom in poly_geoms], dtype=float)
        poly_id_to_idx = {id(g): i for i, g in enumerate(poly_geoms)}
    else:
        poly_index = None
        poly_areas = np.array([])
        poly_id_to_idx = {}

    n = len(streets)
    green_poly_scores = np.zeros(n, dtype=float)
    green_tree_scores = np.zeros(n, dtype=float)

    for i, mid in enumerate(midpoints):
        if mid is None or mid.is_empty:
            continue

        buf = mid.buffer(buffer_distance)

        # ---- Trees contribution ----
        if tree_index is not None:
            cand_ids = tree_index.query(buf)  # indices or geometries
            if len(cand_ids) > 0:
                if isinstance(cand_ids[0], (int, np.integer)):
                    cand_tree_geoms = [tree_geoms[j] for j in cand_ids]
                else:
                    cand_tree_geoms = cand_ids

                dists = np.array([mid.distance(g) for g in cand_tree_geoms], dtype=float)
                mask = dists <= buffer_distance
                dists = dists[mask]
                if dists.size > 0:
                    weights = np.exp(-dists / D)
                    green_tree_scores[i] = weights.sum()

        # ---- Green polygons contribution ----
        if poly_index is not None:
            cand_ids = poly_index.query(buf)  # indices or geometries
            if len(cand_ids) > 0:
                if isinstance(cand_ids[0], (int, np.integer)):
                    poly_idx_arr = np.array(cand_ids, dtype=int)
                    cand_poly_geoms = [poly_geoms[j] for j in poly_idx_arr]
                else:
                    cand_poly_geoms = cand_ids
                    poly_idx_arr = np.array(
                        [poly_id_to_idx[id(g)] for g in cand_poly_geoms],
                        dtype=int,
                    )

                dists = np.array([mid.distance(poly) for poly in cand_poly_geoms], dtype=float)
                mask = dists <= buffer_distance
                if mask.any():
                    dists = dists[mask]
                    poly_idx_arr = poly_idx_arr[mask]
                    area_scaled = poly_areas[poly_idx_arr] / area_scale_m2
                    weights = np.exp(-dists / D) * area_scaled
                    green_poly_scores[i] = weights.sum()

    # Raw scores
    streets["green_raw_poly"] = green_poly_scores
    streets["green_raw_tree"] = green_tree_scores
    streets["green_raw"] = streets["green_raw_poly"] + streets["green_raw_tree"]

    # Non-linear transform to reduce skew, then min–max scaling
    streets["green_raw_log"] = np.log1p(streets["green_raw"])

    gmin = streets["green_raw_log"].min()
    gmax = streets["green_raw_log"].max()
    if np.isfinite(gmin) and np.isfinite(gmax) and gmax > gmin:
        streets["green_index"] = (streets["green_raw_log"] - gmin) / (gmax - gmin)
    else:
        streets["green_index"] = 0.0

    # Clean temporary columns
    streets = streets.drop(columns=["__midpoint", "green_raw_log"])

    return streets

street_lines_green = compute_green_index(
    streets=street_lines,
    green_areas=green_areas,
    trees=trees,
    crs=local_CRS,      # ensures all 3 layers are in the local CRS
    D=100,              # same default as greenR
    buffer_distance=120 # same default as greenR
)

street_lines = street_lines_green

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of the green_index (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='green_index', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['green_index'].min(), street_lines_3857['green_index'].max())
sm.set_array(street_lines_3857['green_index'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

furniture_tags = {
    "amenity": [
        "bench",
        "waste_basket",
        "drinking_water",
        "fountain",
        "recycling",
        "shelter",
        "bicycle_parking"
    ],
    "highway": [
        "bus_stop",
        "street_lamp",
        "crossing"
    ]
}

# Retrieve furniture from OSM (nodes/ways/relations)
furniture = ox.features_from_place(study_area, tags=furniture_tags)
if not isinstance(furniture, gpd.GeoDataFrame):
    furniture = gpd.GeoDataFrame(furniture)

# Classifying furniture types
def classify_furniture(row):
    if 'amenity' in row and pd.notna(row['amenity']):
        if row['amenity'] == 'bench':
            return 'bench'
        elif row['amenity'] == ['waste_basket']:
            return 'waste_basket'
        elif row['amenity'] == 'drinking_water':
            return 'drinking_water'
        elif row['amenity'] == 'fountain':
            return 'fountain'
        elif row['amenity'] == 'recycling':
            return 'recycling'
        elif row['amenity'] == 'shelter':
            return 'shelter'
        elif row['amenity'] == 'bicycle_parking':
            return 'bicycle_parking'
    if 'highway' in row and pd.notna(row['highway']):
        if row['highway'] == 'bus_stop':
            return 'bus_stop'
        elif row['highway'] == 'street_lamp':
            return 'street_lamp'
        elif row['highway'] == 'crossing':
            return 'crossing'
    return np.nan

furniture['furniture_type'] = furniture.apply(classify_furniture, axis=1)

# Setting a filter to only allow "highway" furniture on motorway/trunk segments
def allowed_furniture(row):
    if row.get("furniture_type") in ["bus_stop", "street_lamp"]:
        return True
    return False

furniture["allowed_furniture"] = furniture.apply(allowed_furniture, axis=1)

# Assign furniture to streets using existing buffer_sizes

furniture_proj = furniture.to_crs(street_lines_buffered.crs).reset_index(drop=True)

joined_furniture = gpd.sjoin(
    furniture_proj,
    street_right,
    how="inner",
    predicate="within"
)

# For each row, keep only allowed furniture for motorway/trunk, else keep all
def filter_furniture(row):
    # use renamed column from the right-hand GeoDataFrame
    if row.get("highway") in ["motorway", "trunk"]:
        return row.get("allowed_furniture", False)
    return True

joined_furniture = joined_furniture[joined_furniture.apply(filter_furniture, axis=1)]

# Group by street_id and aggregate furniture types as a list
furniture_by_street = joined_furniture.groupby("street_id")["furniture_type"].agg(list)

# Assign to street_lines as a new column
street_lines["furniture_types"] = street_lines["street_id"].map(furniture_by_street).apply(lambda x: x if isinstance(x, list) else [])

# Calculating urban furniture index per street segment
# 1) Count furniture items per 100 meters of street length
def furniture_per_100m(row):
    furniture_count = len(row["furniture_types"]) if isinstance(row["furniture_types"], list) else 0
    length_m = row["length"] if row["length"] else 1  # avoid division by zero
    count = furniture_count / (length_m / 100)
    return count

street_lines["furniture_per_100m"] = street_lines.apply(furniture_per_100m, axis=1)
street_lines["furniture_per_100m"] = np.arcsinh(street_lines["furniture_per_100m"])

# Winsorize values above 95th percentile for short streets (length <= 30)
p95 = np.percentile(street_lines["furniture_per_100m"].dropna(), 95)
mask = (street_lines["length"] <= 30) & (street_lines["furniture_per_100m"] > p95)
street_lines.loc[mask, "furniture_per_100m"] = p95

street_lines["furniture_per_100m_norm"] = street_lines["furniture_per_100m"] / street_lines["furniture_per_100m"].max()

# 2) Diversity of furniture types using "shannon_entropy" custom function
street_lines["furniture_diversity"] = street_lines["furniture_types"].apply(shannon_entropy)
street_lines["furniture_diversity"] = street_lines["furniture_diversity"] / street_lines["furniture_diversity"].max()

# 3) Composite urban furniture index (geometric mean of normalized count and diversity)
street_lines["furniture_index"] = ((street_lines["furniture_per_100m_norm"] + 1e-9) ** 0.5) * ((street_lines["furniture_diversity"] + 1e-9) ** 0.5)
street_lines["furniture_index"] = street_lines["furniture_index"] / street_lines["furniture_index"].max()

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of the furniture_index (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='furniture_index', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['furniture_index'].min(), street_lines_3857['furniture_index'].max())
sm.set_array(street_lines_3857['furniture_index'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

# Defining a highway level from the perspective of place quality
street_lines["highway_level_place"] = street_lines["highway"].map({
    "pedestrian": 8,
    "living_street": 7,
    "residential": 6,
    "unclassified": 6,
    "tertiary": 5,
    "tertiary_link": 5,
    "secondary": 4,
    "secondary_link": 4,
    "primary": 3,
    "primary_link": 3,
    "trunk": 2,
    "trunk_link": 2,
    "motorway": 1,
    "motorway_link": 1
})

# 0-1 normalization
max_highway_level_place = street_lines["highway_level_place"].max()
min_highway_level_place = street_lines["highway_level_place"].min()
street_lines["highway_level_place"] = (street_lines["highway_level_place"] - min_highway_level_place) / (max_highway_level_place - min_highway_level_place)

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of "highway_level_place" (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='highway_level_place', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['highway_level_place'].min(), street_lines_3857['highway_level_place'].max())
sm.set_array(street_lines_3857['highway_level_place'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

# MIN-MAX Normalizing street_lines["maxspeed"] (from a place perspective)
street_lines["maxspeed_design_norm"] = (street_lines["maxspeed"] - street_lines["maxspeed"].min()) / (street_lines["maxspeed"].max() - street_lines["maxspeed"].min())

# Inverting the scale so that lower speeds have higher values
street_lines["maxspeed_design_norm"] = 1.0 - street_lines["maxspeed_design_norm"]

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of "maxspeed_design_norm" (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='maxspeed_design_norm', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['maxspeed_design_norm'].min(), street_lines_3857['maxspeed_design_norm'].max())
sm.set_array(street_lines_3857['maxspeed_design_norm'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

# Use only the design columns for CRITIC
design_cols = ["green_index", "furniture_index", "highway_level_place", "maxspeed_design_norm"]

# prepare data (drop rows with NA in any of the chosen criteria)
critic_data = street_lines[design_cols].copy()
critic_data = critic_data.apply(pd.to_numeric, errors='coerce')
critic_data_nonan = critic_data.dropna()

if critic_data_nonan.empty:
    raise RuntimeError("No rows without NA in the selected columns; CRITIC cannot be computed.")

# 1) Min-max normalization to [0,1]
mins = critic_data_nonan.min()
maxs = critic_data_nonan.max()
ranges = (maxs - mins).replace(0, np.nan)  # avoid div by zero
norm = (critic_data_nonan - mins) / ranges
norm = norm.fillna(0)

# 2) Standard deviation of each normalized criterion (contrast intensity)
std_dev = norm.std(ddof=0)  # population std

# 3) Correlation matrix of normalized criteria
R = norm.corr()

# 4) Contrast measure C_j = std_j * sum_k (1 - r_jk)
contrast = std_dev * (1 - R).sum(axis=1)

# 5) CRITIC weights (normalized)
critic_weights = contrast / contrast.sum()

# 6) Composite DESIGN_CRITIC score (weighted sum), then min-max to [0,1]
design_raw = (norm * critic_weights).sum(axis=1)
if design_raw.max() > design_raw.min():
    design_norm = (design_raw - design_raw.min()) / (design_raw.max() - design_raw.min())
else:
    design_norm = design_raw * 0.0

# 7) Map results back into street_lines (NaN for rows that were dropped)
street_lines["DESIGN_CRITIC"] = np.nan
street_lines.loc[design_norm.index, "DESIGN_CRITIC"] = design_norm
# Expose weights as a pandas Series for inspection
critic_weights = critic_weights.astype(float)
print("Design CRITIC weights:")
print(critic_weights)

# Compute final_design as a CRITIC-weighted composite of the available normalized/place metrics
# use 'critic_weights' produced by the CRITIC step (was named critic_weights in prior cell)
design_cols = list(critic_weights.index)  # CRITIC criteria names
existing = [c for c in design_cols if c in street_lines.columns]

if not existing:
    raise RuntimeError(f"No CRITIC columns found in street_lines. Expected one of: {design_cols}")
weights = critic_weights.reindex(existing).astype(float)

# Ensure numeric and compute weighted sum while handling NaNs by reweighting per-row
vals = street_lines[existing].apply(pd.to_numeric, errors='coerce')
numerator = (vals * weights).sum(axis=1)
weights_present = vals.notna().astype(float).multiply(weights, axis=1).sum(axis=1)

final_raw = numerator.divide(weights_present).where(weights_present > 0, np.nan)

# Min-max normalize final score to [0,1]
min_final_raw = final_raw.min(skipna=True)
max_final_raw = final_raw.max(skipna=True)
if pd.isna(min_final_raw) or pd.isna(max_final_raw) or max_final_raw <= min_final_raw:
    street_lines["final_design"] = final_raw
else:
    street_lines["final_design"] = (final_raw - min_final_raw) / (max_final_raw - min_final_raw)

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of "final_design" (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='final_design', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['final_design'].min(), street_lines_3857['final_design'].max())
sm.set_array(street_lines_3857['final_design'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

cols = [
    "poi_density_normalized",
    "poi_shannon_entropy",
    "green_index",
    "furniture_index",
    "highway_level_place",
    "maxspeed_design_norm",
]

# Pearson correlation (pairwise, excludes NA)
corr = street_lines[cols].corr(method="pearson")
display(corr.round(4))

# Heatmap visualization
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(cols)

# Annotate cells
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.5 else "black")

fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()

# Using the PLACE columns for CRITIC
place_cols = ["poi_density_normalized", "poi_shannon_entropy", "final_design"]

# prepare data (drop rows with NA in any of the chosen criteria)
place_critic_data = street_lines[place_cols].copy()
place_critic_data = place_critic_data.apply(pd.to_numeric, errors='coerce')
place_critic_data_nonan = place_critic_data.dropna()

if place_critic_data_nonan.empty:
    raise RuntimeError("No rows without NA in the selected columns; CRITIC cannot be computed.")

# 1) Min-max normalization to [0,1]
mins = place_critic_data_nonan.min()
maxs = place_critic_data_nonan.max()
ranges = (maxs - mins).replace(0, np.nan)  # avoid div by zero
norm = (place_critic_data_nonan - mins) / ranges
norm = norm.fillna(0)

# 2) Standard deviation of each normalized criterion (contrast intensity)
std_dev = norm.std(ddof=0)  # population std

# 3) Correlation matrix of normalized criteria
R = norm.corr()

# 4) Contrast measure C_j = std_j * sum_k (1 - r_jk)
contrast = std_dev * (1 - R).sum(axis=1)

# 5) PLACE CRITIC weights (normalized)
place_critic_weights = contrast / contrast.sum()

# 6) Composite PLACE_CRITIC score (weighted sum), then min-max to [0,1]
place_raw = (norm * place_critic_weights).sum(axis=1)
if place_raw.max() > place_raw.min():
    place_norm = (place_raw - place_raw.min()) / (place_raw.max() - place_raw.min())
else:
    place_norm = place_raw * 0.0

# 7) Map results back into street_lines (NaN for rows that were dropped)
street_lines["PLACE_CRITIC"] = np.nan
street_lines.loc[place_norm.index, "PLACE_CRITIC"] = place_norm
# Expose weights as a pandas Series for inspection
place_critic_weights = place_critic_weights.astype(float)
print("PLACE CRITIC weights:")
print(place_critic_weights)

# Compute final PLACE as a CRITIC-weighted composite of the available normalized/place metrics
# use the PLACE CRITIC weights computed previously: `place_critic_weights`
place_cols = list(place_critic_weights.index)  # CRITIC criteria names (e.g. poi_density_normalized, poi_shannon_entropy, ...)
existing = [c for c in place_cols if c in street_lines.columns]

if not existing:
    raise RuntimeError(f"No CRITIC columns found in street_lines. Expected one of: {place_cols}")
weights = place_critic_weights.reindex(existing).astype(float)

# Ensure numeric and compute weighted sum while handling NaNs by reweighting per-row
vals = street_lines[existing].apply(pd.to_numeric, errors='coerce')
numerator = (vals * weights).sum(axis=1)
weights_present = vals.notna().astype(float).multiply(weights, axis=1).sum(axis=1)

final_raw = numerator.divide(weights_present).where(weights_present > 0, np.nan)

# Min-max normalize final score to [0,1] (handle constant/empty cases safely)
min_final_raw = final_raw.min(skipna=True)
max_final_raw = final_raw.max(skipna=True)
if pd.isna(min_final_raw) or pd.isna(max_final_raw) or max_final_raw <= min_final_raw:
    # cannot normalize; keep raw (NaNs preserved)
    street_lines["PLACE"] = final_raw
else:
    street_lines["PLACE"] = (final_raw - min_final_raw) / (max_final_raw - min_final_raw)

# Setting the colors
cmap = LinearSegmentedColormap.from_list("placegrad", place_palette_grad, N=256)
# color for failed values in grey (applied at plotting time)
failed_color = "#808080"
def _failed_mask_from_gdf(gdf):
        return gdf.get('failed_data', False).fillna(False).astype(bool)

failed_mask = _failed_mask_from_gdf(street_lines_3857)

# Matplotlib map of "PLACE" (disable geopandas automatic colorbar)
street_lines_3857 = street_lines.to_crs(epsg=3857)
fig, ax = plt.subplots(figsize=(15, 15))
street_lines_3857[~failed_mask].plot(column='PLACE', ax=ax, legend=False, cmap=cmap, linewidth=1)
street_lines_3857[failed_mask].plot(ax=ax, color=failed_color, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# create a ScalarMappable for a custom, shorter colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(street_lines_3857['PLACE'].min(), street_lines_3857['PLACE'].max())
sm.set_array(street_lines_3857['PLACE'])
# shrink controls the colorbar length (e.g. 0.5 = 50% of original)
cbar = fig.colorbar(sm, ax=ax, fraction=0.0427)
cbar.ax.tick_params(labelsize=10)

# Add a scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

# choose an appropriate scalebar length from a set of "nice" lengths
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15  # aim for ~15% of map width
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

# scalebar position (leave small margins)
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

# draw main bar
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
# end ticks
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
# label centered above the bar
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
        horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.show()

# Histogram of PLACE
plt.figure(figsize=(18, 5))
plt.hist(street_lines['PLACE'], bins=100, color="#4BFF33", edgecolor='white')
plt.xlabel('bins = 100', labelpad=10, size=14)
plt.show()

# Descriptive statistics for street_lines["PLACE"]
s = street_lines["PLACE"].dropna()

measures = ["Count", "Mode", "Mean", "Std. Dev.", "Median", "P25", "P75", "P90", "Skewness", "Kurtosis"]

if s.empty:
    vals = [0, None] + [np.nan] * (len(measures) - 2)
else:
    mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
    vals = [
        int(s.count()),
        mode_val,
        s.mean(),
        s.std(),
        s.median(),
        s.quantile(0.25),
        s.quantile(0.75),
        s.quantile(0.90),
        s.skew(),
        s.kurtosis()
    ]

df_place_stats = pd.DataFrame({"Measure": measures, "Value": vals})
df_place_stats

# --- Absolute classification helper (left-closed, right-open; exact edges go to the upper bin) ---
_eps = np.nextafter(1.0, np.inf)  # slightly above 1.0 so 1.0 is captured in the last bin

abs_bins = [0.0, 0.2, 0.4, 0.6, 0.8, _eps]  # [0,0.2) [0.2,0.4) ... [0.8,1.0]
link_labels = pd.CategoricalDtype(categories=["V", "IV", "III", "II", "I"], ordered=True)
place_labels = pd.CategoricalDtype(categories=["E", "D", "C", "B", "A"], ordered=True)

# Classify LINK and PLACE separately
street_lines["LINK_abs"] = pd.cut(
    street_lines["LINK"],
    bins=abs_bins,
    right=False,                # left-closed, right-open -> boundaries go to the upper bin
    labels=link_labels.categories
).astype(link_labels)

street_lines["PLACE_abs"] = pd.cut(
    street_lines["PLACE"],
    bins=abs_bins,
    right=False,
    labels=place_labels.categories
).astype(place_labels)

# Combine into LP_class_abs
street_lines["LP_class_abs"] = street_lines["LINK_abs"].astype(str) + "-" + street_lines["PLACE_abs"].astype(str)

# --- Relative classification using percentile ranks with ties -> upper class ---
# Percentile rank in (0, 1], using method='average' to preserve order with ties (only "right=False" controls upper binning of ties)
link_prank = street_lines["LINK"].rank(pct=True, method="average")
place_prank = street_lines["PLACE"].rank(pct=True, method="average")

_eps = np.nextafter(1.0, np.inf)  # ensure rank==1.0 lands in last bin
q_bins = [0.0, 0.2, 0.4, 0.6, 0.8, _eps]  # quintile cutpoints; boundaries go to the upper bin

link_labels = pd.CategoricalDtype(categories=["V", "IV", "III", "II", "I"], ordered=True)
place_labels = pd.CategoricalDtype(categories=["E", "D", "C", "B", "A"], ordered=True)

street_lines["LINK_rel"] = pd.cut(
    link_prank,
    bins=q_bins,
    right=False,  # boundary values (including ties that land exactly on a cut) go to the upper bin
    labels=link_labels.categories
).astype(link_labels)

street_lines["PLACE_rel"] = pd.cut(
    place_prank,
    bins=q_bins,
    right=False,
    labels=place_labels.categories
).astype(place_labels)

# Combine into LP_class_rel
street_lines["LP_class_rel"] = street_lines["LINK_rel"].astype(str) + "-" + street_lines["PLACE_rel"].astype(str)

# Histograms of LINK typologies (absolute and relative)

order = ["V", "IV", "III", "II", "I"]
counts_abs = street_lines['LINK_abs'].value_counts().reindex(order).fillna(0)
counts_rel = street_lines['LINK_rel'].value_counts().reindex(order).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Absolute typology
bars0 = axes[0].bar(order, counts_abs.values, color="#FF3333", edgecolor="white")
axes[0].set_title("Absolute Classification Typologies")
axes[0].set_xticks(range(len(order)))
axes[0].set_xticklabels(order)

# Relative typology
bars1 = axes[1].bar(order, counts_rel.values, color="#FF3333", edgecolor="white")
axes[1].set_title("Relative Classification Typologies")
axes[1].set_xticks(range(len(order)))
axes[1].set_xticklabels(order)

# Add value labels above each bar with count and percentage
total_abs = counts_abs.sum()
total_rel = counts_rel.sum()
ymax = max(counts_abs.max(), counts_rel.max())
offset = ymax * 0.01 if ymax > 0 else 0.1

for rect, val in zip(bars0, counts_abs.values):
    pct = (val / total_abs * 100) if total_abs > 0 else 0
    axes[0].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + offset, f"{int(val)} ({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

for rect, val in zip(bars1, counts_rel.values):
    pct = (val / total_rel * 100) if total_rel > 0 else 0
    axes[1].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + offset, f"{int(val)} ({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Categorical map with legend for LINK_abs

street_lines_3857 = street_lines.to_crs(epsg=3857)
# map colors (missing -> grey)
# build colors as object dtype to avoid assigning new category into an existing Categorical column
colors = street_lines_3857['LINK_abs'].map(link_palette_cat)
colors = colors.astype(object).fillna('#808080')

street_lines_3857 = street_lines_3857.copy()
street_lines_3857['cat_color'] = colors
fig, ax = plt.subplots(figsize=(10, 10))
street_lines_3857.plot(color=street_lines_3857['cat_color'], ax=ax, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Legend (ordered from V -> I) — enlarged
order = ["V", "IV", "III", "II", "I"]
handles = [mpatches.Patch(color=link_palette_cat[k], label=k) for k in order]
# include a patch for missing/failed data
handles.append(mpatches.Patch(color='#808080', label='Null'))

legend = ax.legend(
        handles=handles,
        loc='upper left',
        frameon=True,
        fontsize=14,
        title_fontsize=16,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=1.8,
        handleheight=1.8
)
legend.get_frame().set_alpha(0.95)

# Scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
                horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

# Categorical map with legend for LINK_rel

street_lines_3857 = street_lines.to_crs(epsg=3857)
# map colors (missing -> grey)
# build colors as object dtype to avoid assigning new category into an existing Categorical column
colors = street_lines_3857['LINK_rel'].map(link_palette_cat)
colors = colors.astype(object).fillna('#808080')

street_lines_3857 = street_lines_3857.copy()
street_lines_3857['cat_color'] = colors
fig, ax = plt.subplots(figsize=(10, 10))
street_lines_3857.plot(color=street_lines_3857['cat_color'], ax=ax, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Legend (ordered from V -> I) — enlarged
order = ["V", "IV", "III", "II", "I"]
handles = [mpatches.Patch(color=link_palette_cat[k], label=k) for k in order]
# include a patch for missing/failed data
handles.append(mpatches.Patch(color='#808080', label='Null'))

legend = ax.legend(
        handles=handles,
        loc='upper left',
        frameon=True,
        fontsize=14,
        title_fontsize=16,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=1.8,
        handleheight=1.8
)
legend.get_frame().set_alpha(0.95)

# Scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
                horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

# Histograms of PLACE typologies (absolute and relative)

order = ["E", "D", "C", "B", "A"]
counts_abs = street_lines['PLACE_abs'].value_counts().reindex(order).fillna(0)
counts_rel = street_lines['PLACE_rel'].value_counts().reindex(order).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Absolute typology
bars0 = axes[0].bar(order, counts_abs.values, color="#33FF33", edgecolor="white")
axes[0].set_title("Absolute Classification Typologies")
axes[0].set_xticks(range(len(order)))
axes[0].set_xticklabels(order)

# Relative typology
bars1 = axes[1].bar(order, counts_rel.values, color="#33FF33", edgecolor="white")
axes[1].set_title("Relative Classification Typologies")
axes[1].set_xticks(range(len(order)))
axes[1].set_xticklabels(order)

# Add value labels above each bar with count and percentage
total_abs = counts_abs.sum()
total_rel = counts_rel.sum()
ymax = max(counts_abs.max(), counts_rel.max())
offset = ymax * 0.01 if ymax > 0 else 0.1

for rect, val in zip(bars0, counts_abs.values):
    pct = (val / total_abs * 100) if total_abs > 0 else 0
    axes[0].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + offset, f"{int(val)} ({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

for rect, val in zip(bars1, counts_rel.values):
    pct = (val / total_rel * 100) if total_rel > 0 else 0
    axes[1].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + offset, f"{int(val)} ({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Categorical map with legend for PLACE_abs

street_lines_3857 = street_lines.to_crs(epsg=3857)
# map colors (missing -> grey)
# build colors as object dtype to avoid assigning new category into an existing Categorical column
colors = street_lines_3857['PLACE_abs'].map(place_palette_cat)
colors = colors.astype(object).fillna('#808080')

street_lines_3857 = street_lines_3857.copy()
street_lines_3857['cat_color'] = colors
fig, ax = plt.subplots(figsize=(10, 10))
street_lines_3857.plot(color=street_lines_3857['cat_color'], ax=ax, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Legend (ordered from V -> I) — enlarged
order = ["E", "D", "C", "B", "A"]
handles = [mpatches.Patch(color=place_palette_cat[k], label=k) for k in order]
# include a patch for missing/failed data
handles.append(mpatches.Patch(color='#808080', label='Null'))

legend = ax.legend(
        handles=handles,
        loc='upper left',
        frameon=True,
        fontsize=14,
        title_fontsize=16,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=1.8,
        handleheight=1.8
)
legend.get_frame().set_alpha(0.95)

# Scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
                horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

# Categorical map with legend for PLACE_rel

street_lines_3857 = street_lines.to_crs(epsg=3857)
# map colors (missing -> grey)
# build colors as object dtype to avoid assigning new category into an existing Categorical column
colors = street_lines_3857['PLACE_rel'].map(place_palette_cat)
colors = colors.astype(object).fillna('#808080')

street_lines_3857 = street_lines_3857.copy()
street_lines_3857['cat_color'] = colors
fig, ax = plt.subplots(figsize=(10, 10))
street_lines_3857.plot(color=street_lines_3857['cat_color'], ax=ax, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis('off')

# Legend (ordered from V -> I) — enlarged
order = ["E", "D", "C", "B", "A"]
handles = [mpatches.Patch(color=place_palette_cat[k], label=k) for k in order]
# include a patch for missing/failed data
handles.append(mpatches.Patch(color='#808080', label='Null'))

legend = ax.legend(
        handles=handles,
        loc='upper left',
        frameon=True,
        fontsize=14,
        title_fontsize=16,
        borderpad=1.2,
        labelspacing=0.8,
        handlelength=1.8,
        handleheight=1.8
)
legend.get_frame().set_alpha(0.95)

# Scale bar (meters) in lower-right corner (data coordinates are in meters for EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny

candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])

x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin

ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle='butt')
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
                horizontalalignment='center', verticalalignment='bottom', fontsize=10, color="#000000")

plt.tight_layout()
plt.show()

# Correlating LINK and PLACE (pairwise, excluding NaNs)
x = street_lines["LINK"]
y = street_lines["PLACE"]
mask = x.notna() & y.notna()
pearson = x[mask].corr(y[mask], method="pearson")

# Testing statistical significance of the Pearson correlation
if mask.sum() > 1:
	stat, p_value = pearsonr(x[mask], y[mask])
else:
	stat, p_value = (np.nan, np.nan)

print(f"n = {mask.sum()}")
print(f"Pearson:  {pearson:.4f}")
print(f"Pearson correlation p-value: {p_value:.4e}")

# Scatter plot of LINK-PLACE with marginal histograms

# Data axes
x = street_lines["PLACE"]
y = street_lines["LINK"]

# Define figure and grid
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(4, 4, wspace=0.05, hspace=0.05)

# === Main scatter plot ===
ax = fig.add_subplot(gs[1:4, 0:3])
# Map LP_class_rel to colors using full_palette_cat
point_colors = street_lines["LP_class_rel"].map(full_palette_cat).fillna("#cccccc")
ax.scatter(x, y, alpha=0.5, s=10, color=point_colors)
ax.set_xlabel("PLACE")
ax.set_ylabel("LINK")
ax.grid(True, linestyle='--', alpha=0.3)

# === Top histogram (x) ===
ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax)
counts_x, bins_x, patches_x = ax_histx.hist(
    x,
    bins=100,
    color='#00ff00',
    alpha=0.7,
    edgecolor='white',
    linewidth=0.3
)
# Hide x-scale (shared with scatter), keep frequency scale
ax_histx.tick_params(axis='x', labelbottom=False, bottom=False)

# Make the tallest bin touch the ceiling: max bin == axis max
max_count_x = counts_x.max()
ax_histx.set_ylim(0, max_count_x)

# Show only half and max of the window (no 0)
ax_histx.set_yticks([max_count_x / 2, max_count_x])
ax_histx.tick_params(axis='y', direction='out', pad=4, labelsize=8)

# === Right histogram (y) ===
ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax)
counts_y, bins_y, patches_y = ax_histy.hist(
    y,
    bins=100,
    orientation='horizontal',
    color='#ff0000',
    alpha=0.7,
    edgecolor='white',
    linewidth=0.3
)
# Hide y-scale (shared with scatter), keep frequency scale
ax_histy.tick_params(axis='y', labelleft=False, left=False)

# Make the tallest bin touch the ceiling: max bin == axis max
max_count_y = counts_y.max()
ax_histy.set_xlim(0, max_count_y)

# Show only half and max of the window (no 0)
ax_histy.set_xticks([max_count_y / 2, max_count_y])
ax_histy.tick_params(axis='x', direction='out', pad=4, labelsize=8)

# Adding a Pearson correlation trendline to the scatter plot
if mask.sum() >= 2:
    # fit only if enough points
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='blue', linestyle='--', linewidth=1)

# Display the Pearson r and the statistical significance on the corner of the scatter plot
ax.text(0.66, 0.97, 
        f"Pearson's r = {pearson:.4f}\nP-value = {p_value:.4e}", 
        transform=ax.transAxes,
        verticalalignment='top', 
        fontsize=10, 
        color='blue')

# === Layout adjustments ===
ax.margins(x=0, y=0)
plt.tight_layout()
plt.show()

# Static map of absolute LP classes with CartoDB Positron basemap and external legend image

street_lines_3857 = street_lines.to_crs(epsg=3857)
colors = street_lines_3857["LP_class_abs"].map(full_palette_cat).astype(object).fillna("#808080")
street_lines_3857 = street_lines_3857.copy()
street_lines_3857["cat_color"] = colors

fig, ax = plt.subplots(figsize=(12, 12))
street_lines_3857.plot(color=street_lines_3857["cat_color"], ax=ax, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis("off")

# scale bar (meters, EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle="butt")
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
    horizontalalignment="center", verticalalignment="bottom", fontsize=10, color="#000000")

legend_img = plt.imread("link_place_classes.png")

imagebox = OffsetImage(
    legend_img,
    zoom=0.1
)

legend_ab = AnnotationBbox(
    imagebox,
    (0.015, 0.985),        # upper-left corner
    xycoords="axes fraction",
    box_alignment=(0, 1),
    frameon=True,
    bboxprops=dict(
        facecolor="white",
        edgecolor="#9e9e9e",
        linewidth=1.0,
        boxstyle="round,pad=0.4,rounding_size=0.2"
    )
)

ax.add_artist(legend_ab)



plt.tight_layout()
plt.show()

# Static map of relative LP classes with CartoDB Positron basemap and external legend image

street_lines_3857 = street_lines.to_crs(epsg=3857)
colors = street_lines_3857["LP_class_rel"].map(full_palette_cat).astype(object).fillna("#808080")
street_lines_3857 = street_lines_3857.copy()
street_lines_3857["cat_color"] = colors

fig, ax = plt.subplots(figsize=(12, 12))
street_lines_3857.plot(color=street_lines_3857["cat_color"], ax=ax, linewidth=1)
study_area_3857.plot(ax=ax, linestyle=":", linewidth=1, edgecolor=(0, 0, 0, 1.0), facecolor=(0, 0, 0, 0))
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.axis("off")

# scale bar (meters, EPSG:3857)
minx, miny, maxx, maxy = street_lines_3857.total_bounds
map_width = maxx - minx
map_height = maxy - miny
candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
target = map_width * 0.15
scalebar_len = max([c for c in candidates if c <= target], default=candidates[0])
x_margin = map_width * 0.01
y_margin = map_height * 0.01
x_end = maxx - x_margin
x_start = x_end - scalebar_len
y = miny + y_margin
ax.plot([x_start, x_end], [y, y], color="#000000", linewidth=2, solid_capstyle="butt")
tick_h = map_height * 0.01
ax.plot([x_start, x_start], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.plot([x_end, x_end], [y - tick_h / 2, y + tick_h / 2], color="#000000", linewidth=2)
ax.text((x_start + x_end) / 2, y + tick_h, f"{int(scalebar_len)} m",
    horizontalalignment="center", verticalalignment="bottom", fontsize=10, color="#000000")

legend_img = plt.imread("link_place_classes.png")

imagebox = OffsetImage(
    legend_img,
    zoom=0.1
)

legend_ab = AnnotationBbox(
    imagebox,
    (0.015, 0.985),        # upper-left corner
    xycoords="axes fraction",
    box_alignment=(0, 1),
    frameon=True,
    bboxprops=dict(
        facecolor="white",
        edgecolor="#9e9e9e",
        linewidth=1.0,
        boxstyle="round,pad=0.4,rounding_size=0.2"
    )
)

ax.add_artist(legend_ab)



plt.tight_layout()
plt.show()

# Interactive folium report

# -----------------------------------------------------------
# Reproject street_lines to WGS84 for folium
# -----------------------------------------------------------
street_lines_4326 = street_lines.to_crs(epsg=4326)

# Ensure study_centroid exists; compute from street_lines_4326 (WGS84) or fallback to study_area_gdf
try:
    study_centroid
except NameError:
    if 'street_lines_4326' in globals() and not street_lines_4326.empty:
        study_centroid = street_lines_4326.unary_union.centroid
    elif 'study_area_gdf' in globals() and not study_area_gdf.empty:
        study_centroid = study_area_gdf.to_crs(epsg=4326).geometry.unary_union.centroid
    else:
        raise NameError(
            "study_centroid is not defined and cannot be computed; provide 'street_lines' or 'study_area_gdf'"
        )

# ----------------------------------------------------------
# Map base
# ----------------------------------------------------------
m_lp = folium.Map(
    location=[study_centroid.y, study_centroid.x],
    zoom_start=12,
    tiles=None
)
folium.TileLayer("CartoDB positron", name="Basemap", control=False).add_to(m_lp)
# ----------------------------------------------------------
# Helper function to get color from colormap
# ----------------------------------------------------------
def _color_from_cmap(cmap, value, vmin, vmax):
    if value is None or not np.isfinite(value):
        return "#cccccc"  # grey for missing/invalid data
    # Normalize value to [0, 1]
    norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    norm_value = min(max(norm_value, 0.0), 1.0)  # clamp to [0, 1]
    return cmap(norm_value)

# ----------------------------------------------------------
# Helper to convert a GeoDataFrame to a GeoJSON-like dict for folium
# ----------------------------------------------------------
def gdf_to_geojson_dict(gdf):
    """
    Return a GeoJSON-like Python dict from a GeoDataFrame or mapping.
    This is robust: it tries to use GeoDataFrame.to_crs(...).to_json(),
    falls back to other representations (to_json, __geo_interface__) and
    always runs a normalizer to convert numpy/pandas scalars/arrays
    into native Python types so json serialization (used by folium) does not fail.
    """
    if gdf is None:
        return {}

    # If already a mapping/dict-like, normalize and return
    if isinstance(gdf, dict):
        return _normalize_numpy_types(gdf)

    # Try the usual GeoDataFrame -> GeoJSON string -> dict path
    try:
        if hasattr(gdf, "to_crs"):
            geojson_str = gdf.to_crs(epsg=4326).to_json()
        else:
            geojson_str = gdf.to_json()
        geojson = json.loads(geojson_str)
        return _normalize_numpy_types(geojson)
    except Exception:
        # Try alternative representations
        try:
            # __geo_interface__ is commonly available on GeoDataFrame/GeoSeries
            geo_interface = getattr(gdf, "__geo_interface__", None)
            if geo_interface is not None:
                return _normalize_numpy_types(geo_interface)
        except Exception:
            pass

        try:
            # Try direct to_json without reproject (if reprojection failed)
            geojson = json.loads(gdf.to_json())
            return _normalize_numpy_types(geojson)
        except Exception:
            pass

    # As a last resort, attempt to convert pandas DataFrame to records
    try:
        import pandas as _pd  # local import to avoid altering module-level imports
        if isinstance(gdf, _pd.DataFrame):
            return _normalize_numpy_types(json.loads(gdf.to_json(orient="records")))
    except Exception:
        pass

    # If nothing worked, raise a helpful error
    raise TypeError("Could not convert provided object to a GeoJSON-like dict")


def _normalize_numpy_types(obj):
    """
    Recursively convert numpy/pandas scalar types and arrays into native Python types.
    Handles numpy scalar types, numpy arrays, pandas scalars (Timestamp, NA),
    pandas Categorical, and nested dict/list structures.
    """
    # dict-like
    if isinstance(obj, dict):
        return {k: _normalize_numpy_types(v) for k, v in obj.items()}

    # list-like / tuple
    if isinstance(obj, (list, tuple)):
        return [_normalize_numpy_types(v) for v in obj]

    # numpy arrays -> convert to list first
    if isinstance(obj, np.ndarray):
        try:
            return _normalize_numpy_types(obj.tolist())
        except Exception:
            # fallback: iterate
            return [_normalize_numpy_types(v) for v in obj]

    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        try:
            if np.isnan(obj):
                return None
        except Exception:
            pass
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # pandas / pyarrow / other library scalars
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        if obj is _pd.NaT:
            return None
        # pandas categorical -> list
        if isinstance(obj, _pd.Categorical):
            return _normalize_numpy_types(obj.tolist())
        # pandas scalar (e.g., Int64 NA-aware) -> use pandas.isna to check
        if hasattr(_pd, "isna") and _pd.isna(obj):
            return None
    except Exception:
        # pandas may not be available; ignore
        pass

    # plain Python float nan check
    try:
        if isinstance(obj, float) and np.isnan(obj):
            return None
    except Exception:
        pass

    # fallback: return as-is (strings, ints, booleans, None)
    return obj

# ----------------------------------------------------------
# Defining "cmap"
# ----------------------------------------------------------
# ----------------------------------------------------------

cmap_link = BrancaLinearColormap(
    colors=link_palette_grad,
    vmin=street_lines["LINK"].min(),
    vmax=street_lines["LINK"].max()
)

cmap_place = BrancaLinearColormap(
    colors=place_palette_grad,
    vmin=street_lines["PLACE"].min(),
    vmax=street_lines["PLACE"].max()
)

link_min = street_lines["LINK"].min()
link_max = street_lines["LINK"].max()

place_min = street_lines["PLACE"].min()
place_max = street_lines["PLACE"].max()

# ----------------------------------------------------------
# LINK layer
# ----------------------------------------------------------
fg_LINK = folium.FeatureGroup(name="LINK", overlay=False, show=True)

folium.GeoJson(
    gdf_to_geojson_dict(street_lines_4326),
    style_function=lambda feat: {
        "color": _color_from_cmap(
            cmap_link,
            feat["properties"].get("LINK"),
            link_min,
            link_max,
        ),
        "weight": 2,
        "opacity": 0.9,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["street_id", "name", "LINK"],
        aliases=["Street ID:", "Name:", "LINK:"],
        localize=True
    )
).add_to(fg_LINK)

fg_LINK.add_to(m_lp)

# ----------------------------------------------------------
# PLACE layer
# ----------------------------------------------------------
fg_PLACE = folium.FeatureGroup(name="PLACE", overlay=False, show=False)

folium.GeoJson(
    gdf_to_geojson_dict(street_lines_4326),
    style_function=lambda feat: {
        "color": _color_from_cmap(
            cmap_place,
            feat["properties"].get("PLACE"),
            place_min,
            place_max,
        ),
        "weight": 2,
        "opacity": 0.9,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["street_id", "name", "PLACE"],
        aliases=["Street ID:", "Name:", "PLACE:"],
        localize=True
    )
).add_to(fg_PLACE)

fg_PLACE.add_to(m_lp)

# ----------------------------------------------------------
# ABS layer
# ----------------------------------------------------------
fg_ABS = folium.FeatureGroup(
    name="Absolute Classification",
    overlay=False,
    show=False
)

folium.GeoJson(
    gdf_to_geojson_dict(street_lines_4326),
    style_function=lambda feat: {
        "color": full_palette_cat.get(
            feat["properties"].get("LP_class_abs"), "#cccccc"
        ),
        "weight": 2,
        "opacity": 0.9,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["street_id","name","LP_class_abs"],
        aliases=["Street ID:","Name:","Absolute Classification:"],
        localize=True
    )
).add_to(fg_ABS)

fg_ABS.add_to(m_lp)

# ----------------------------------------------------------
# REL layer
# ----------------------------------------------------------
fg_REL = folium.FeatureGroup(
    name="Relative Classification",
    overlay=False,
    show=False
)

folium.GeoJson(
    gdf_to_geojson_dict(street_lines_4326),
    style_function=lambda feat: {
        "color": full_palette_cat.get(
            feat["properties"].get("LP_class_rel"), "#cccccc"
        ),
        "weight": 2,
        "opacity": 0.9,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["street_id","name","LP_class_rel"],
        aliases=["Street ID:","Name:","Relative Classification:"],
        localize=True
    )
).add_to(fg_REL)

fg_REL.add_to(m_lp)

# ----------------------------------------------------------
# Legend HTML containers
# ----------------------------------------------------------
link_box = f"""
<div class="layer-colorbar" id="cb_LINK"
     style="position:absolute;bottom:10px;left:10px;
            z-index:9999;background:white;padding:4px;
            border-radius:4px;">
    <img src="link_gradient_colormap.png" width="350">
</div>
"""
m_lp.get_root().html.add_child(Element(link_box))

place_box = f"""
<div class="layer-colorbar" id="cb_PLACE"
     style="position:absolute;bottom:10px;left:10px;
            z-index:9999;background:white;padding:4px;
            border-radius:4px;display:none;">
    <img src="place_gradient_colormap.png" width="350">
</div>
"""
m_lp.get_root().html.add_child(Element(place_box))

abs_box = """
<div class="layer-colorbar" id="cb_Absolute_Classification"
     style="position:absolute;bottom:10px;left:10px;
            z-index:9999;background:white;padding:4px;
            border-radius:4px;display:none;">
  <img src="link_place_classes.png" width="250">
</div>
"""
m_lp.get_root().html.add_child(Element(abs_box))

rel_box = """
<div class="layer-colorbar" id="cb_Relative_Classification"
     style="position:absolute;bottom:10px;left:10px;
            z-index:9999;background:white;padding:4px;
            border-radius:4px;display:none;">
  <img src="link_place_classes.png" width="250">
</div>
"""
m_lp.get_root().html.add_child(Element(rel_box))

# ----------------------------------------------------------
# Layer control
# ----------------------------------------------------------
folium.LayerControl(collapsed=False).add_to(m_lp)

# ----------------------------------------------------------
# JS legend toggle logic
# ----------------------------------------------------------
js = """
<script>

$(document).ready(() => {
    $(".leaflet-control-layers-list").change((e) => {
        // Start by hiding them all
        $(".layer-colorbar").fadeOut();
        // Get value of selected option
        let val = $(e.target).parent().find("span").text();
        console.log("Changed to", val);
        // Show corresponding colorbar
        if (val.includes("LINK")) {
            $("#cb_LINK").fadeIn();
            return;
        } else if (val.includes("PLACE")) {
            $("#cb_PLACE").fadeIn();
            return;
        } else if (val.includes("Absolute")) {
            $("#cb_Absolute_Classification").fadeIn();
            return;
        } else if (val.includes("Relative")) {
            $("#cb_Relative_Classification").fadeIn();
            return;
        }
    });
});

</script>
"""
m_lp.get_root().html.add_child(Element(js))

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
m_lp.save("lp_folium_map.html")

