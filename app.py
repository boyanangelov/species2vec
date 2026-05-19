"""Interactive species2vec demo (Streamlit).

Run:
    streamlit run app.py

Two embedding models side-loaded from runs/compare:
- fixed.vec   — char n-grams disabled, geohash p5, one sentence per bin
- broken.vec  — original recipe (n-grams on, one big string corpus)

Switch between them to see the name-leak in action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from gensim.models import KeyedVectors


REPO = Path(__file__).resolve().parent
VEC_FIXED = REPO / "runs" / "compare" / "fixed.vec"
VEC_BROKEN = REPO / "runs" / "compare" / "broken.vec"
CSV = REPO / "data" / "squamata.csv"
UMAP_CACHE = REPO / "runs" / "umap_cache"
UMAP_CACHE.mkdir(parents=True, exist_ok=True)
VERNACULAR_CACHE = REPO / "runs" / "vernacular.json"

def _basemap_spec(name: str) -> tuple[str, list[dict]]:
    """Translate a sidebar choice into (map_style, map_layers) for Plotly.

    For relief / topo tilesets, return a transparent base + a raster layer
    pointing at a public XYZ tile source. No API key needed.
    """
    if name == "relief (OpenTopoMap)":
        return "white-bg", [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "© OpenTopoMap (CC-BY-SA), SRTM",
                "source": ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png"],
            }
        ]
    if name == "relief (ESRI shaded)":
        return "white-bg", [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Tiles © Esri — Source: Esri",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ]
    if name == "topo (ESRI World Topo)":
        return "white-bg", [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Tiles © Esri — Source: Esri, USGS, NOAA",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ]
    return name, []


def _auto_view(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float, int]:
    """Center + zoom that fits the bounding box of the data."""
    lat_c = float((np.nanmax(lats) + np.nanmin(lats)) / 2)
    lon_c = float((np.nanmax(lons) + np.nanmin(lons)) / 2)
    span = float(max(np.nanmax(lats) - np.nanmin(lats), np.nanmax(lons) - np.nanmin(lons)))
    if span < 1:
        z = 7
    elif span < 5:
        z = 5
    elif span < 20:
        z = 3
    elif span < 60:
        z = 2
    else:
        z = 1
    return lat_c, lon_c, z


EARTH_R_KM = 6371.0088


def _load_vernacular_cache() -> dict[str, str]:
    if VERNACULAR_CACHE.exists():
        try:
            return json.loads(VERNACULAR_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_vernacular_cache(d: dict[str, str]) -> None:
    VERNACULAR_CACHE.write_text(json.dumps(d, sort_keys=True))


@st.cache_data(show_spinner=False)
def vernacular_for(species_underscored: str) -> str:
    """Look up an English common name from GBIF. Disk-cached across runs."""
    cache = _load_vernacular_cache()
    if species_underscored in cache:
        return cache[species_underscored]
    sci = species_underscored.replace("_", " ")
    name = ""
    try:
        m = requests.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": sci, "kingdom": "Animalia"},
            timeout=4,
        ).json()
        key = m.get("usageKey")
        if key:
            v = requests.get(
                f"https://api.gbif.org/v1/species/{key}/vernacularNames",
                params={"limit": 50},
                timeout=4,
            ).json()
            results = v.get("results", [])
            eng = [r["vernacularName"] for r in results if r.get("language") == "eng"]
            if eng:
                name = eng[0]
            elif results:
                name = results[0].get("vernacularName", "")
    except (requests.RequestException, ValueError):
        pass
    cache[species_underscored] = name
    _save_vernacular_cache(cache)
    return name


def _label(species_underscored: str) -> str:
    sci = species_underscored.replace("_", " ")
    common = vernacular_for(species_underscored)
    return f"{sci} ({common})" if common else sci


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


@st.cache_resource
def load_vec(path: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(path)


@st.cache_data
def load_occurrences(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["species", "decimalLatitude", "decimalLongitude"])
    df["species"] = df["species"].astype(str).str.strip().str.replace(" ", "_", regex=False)
    return df


@st.cache_data
def species_centroids(df: pd.DataFrame, min_occ: int = 5) -> pd.DataFrame:
    lat = np.radians(df["decimalLatitude"].to_numpy())
    lon = np.radians(df["decimalLongitude"].to_numpy())
    d = df[["species"]].copy()
    d["x"] = np.cos(lat) * np.cos(lon)
    d["y"] = np.cos(lat) * np.sin(lon)
    d["z"] = np.sin(lat)
    g = cast(
        pd.DataFrame,
        d.groupby("species").agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean")),
    )
    g = cast(pd.DataFrame, g.loc[g["n"] >= min_occ])
    norm = np.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)
    g["lat"] = np.degrees(np.arcsin(g["z"] / norm))
    g["lon"] = np.degrees(np.arctan2(g["y"], g["x"]))
    return cast(pd.DataFrame, g[["lat", "lon", "n"]].reset_index())


@st.cache_data
def compute_umap(vec_path: str, seed: int = 0) -> pd.DataFrame:
    import umap

    cache = UMAP_CACHE / (Path(vec_path).stem + f"_seed{seed}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)

    kv = load_vec(vec_path)
    words = list(kv.key_to_index.keys())
    V = np.stack([kv[w] for w in words])
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(V)
    out = pd.DataFrame({"species": words, "u1": coords[:, 0], "u2": coords[:, 1]})
    out.to_parquet(cache)
    return out


def top_k_neighbors(kv: KeyedVectors, species: str, k: int) -> pd.DataFrame:
    sims = kv.most_similar(species, topn=k)
    return pd.DataFrame(sims, columns=pd.Index(["species", "cosine"]))


def main() -> None:
    st.set_page_config(page_title="species2vec demo", layout="wide")
    st.title("species2vec — what does the embedding actually learn?")
    st.caption(
        "147 k Squamata (reptile) occurrences from GBIF, 100-d fastText embeddings. "
        "Toggle **broken** vs **fixed** to see character-n-gram name leakage in action."
    )

    with st.sidebar:
        st.header("Controls")
        model_label = st.radio(
            "Embedding",
            ["fixed (no n-grams)", "broken (n-grams on)"],
            help="The 'broken' model is the recipe shipped on bioRxiv. The 'fixed' model "
            "disables fastText character n-grams so the model cannot read taxonomy off names.",
        )
        vec_path = str(VEC_FIXED if model_label.startswith("fixed") else VEC_BROKEN)
        k = st.slider("Top-K nearest species", 5, 30, 12)
        max_occ = st.slider("Max occurrences plotted per species", 50, 2000, 400, step=50)
        basemap = st.selectbox(
            "Basemap",
            [
                "relief (OpenTopoMap)",
                "relief (ESRI shaded)",
                "topo (ESRI World Topo)",
                "carto-voyager",
                "open-street-map",
                "carto-positron",
                "carto-darkmatter",
                "white-bg",
            ],
            index=0,
            help="Relief / topo tilesets use OpenTopoMap and ESRI; no API key required.",
        )
        st.markdown("---")
        st.caption("Each marker on the map is one GBIF record. Colors group species.")

    kv = load_vec(vec_path)
    occ = load_occurrences(str(CSV))
    cent = species_centroids(occ)
    umap_df = compute_umap(vec_path)

    vocab_in_occ = sorted(set(kv.key_to_index) & set(cent["species"]))
    if not vocab_in_occ:
        st.error("No overlap between vector vocab and occurrence species.")
        return
    default_idx = vocab_in_occ.index("Pogona_vitticeps") if "Pogona_vitticeps" in vocab_in_occ else 0
    species = st.selectbox("Focus species", vocab_in_occ, index=default_idx)

    neigh = top_k_neighbors(kv, species, k)
    sel_centroid = cent.loc[cent["species"] == species]
    if sel_centroid.empty:
        st.warning(f"{species} has fewer than 5 occurrences.")
        return
    sel_lat = float(sel_centroid["lat"].iloc[0])
    sel_lon = float(sel_centroid["lon"].iloc[0])

    neigh = neigh.merge(cent, on="species", how="left")
    neigh["km_to_focus"] = _haversine_km(sel_lat, sel_lon, neigh["lat"], neigh["lon"])
    neigh["same_genus"] = neigh["species"].str.split("_").str[0] == species.split("_")[0]

    leak_frac = float(neigh["same_genus"].mean())
    mean_km = float(neigh["km_to_focus"].dropna().mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Focus species", _label(species))
    c2.metric("Vocab size", f"{len(kv.key_to_index):,}")
    c3.metric(
        "Same-genus in top-K",
        f"{leak_frac:.0%}",
        help="High % on broken, lower on fixed = char-n-gram leak vs spatial signal.",
    )
    c4.metric(
        "Mean km to neighbors",
        f"{mean_km:,.0f} km" if not np.isnan(mean_km) else "n/a",
    )

    st.markdown("### Map of occurrences — focus species + top-K neighbors")
    plot_species = [species] + neigh["species"].tolist()
    sub = occ[occ["species"].isin(plot_species)].copy()
    sub = pd.concat(
        [
            g.sample(min(len(g), max_occ), random_state=0)
            for _, g in sub.groupby("species", sort=False)
        ],
        ignore_index=True,
    )
    sub["role"] = np.where(sub["species"] == species, "focus", "neighbor")

    lat_c, lon_c, zoom = _auto_view(
        cast(pd.Series, sub["decimalLatitude"]).to_numpy(),
        cast(pd.Series, sub["decimalLongitude"]).to_numpy(),
    )
    base_style, base_layers = _basemap_spec(basemap)
    fig_map = px.scatter_map(
        sub,
        lat="decimalLatitude",
        lon="decimalLongitude",
        color="species",
        opacity=0.75,
        hover_name="species",
        zoom=zoom,
        center={"lat": lat_c, "lon": lon_c},
        map_style=base_style,
        height=600,
    )
    fig_map.update_traces(marker=dict(size=7))
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        map=dict(layers=base_layers) if base_layers else {},
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("### Embedding space (UMAP of all species vectors)")
    umap_df = umap_df.copy()
    umap_df["role"] = "other"
    umap_df.loc[umap_df["species"] == species, "role"] = "focus"
    umap_df.loc[umap_df["species"].isin(neigh["species"]), "role"] = "neighbor"
    umap_df["genus"] = umap_df["species"].str.split("_").str[0]
    umap_df["same_genus"] = umap_df["genus"] == species.split("_")[0]
    color_map = {"other": "#cccccc", "neighbor": "#1f77b4", "focus": "#d62728"}
    umap_df["size"] = umap_df["role"].replace({"other": 3, "neighbor": 9, "focus": 18}).astype(float)

    fig_umap = px.scatter(
        umap_df,
        x="u1",
        y="u2",
        color="role",
        size="size",
        size_max=18,
        color_discrete_map=color_map,
        hover_name="species",
        hover_data={"u1": False, "u2": False, "role": True, "size": False, "genus": True},
        height=520,
    )
    fig_umap.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
    )
    st.plotly_chart(fig_umap, use_container_width=True)
    st.caption(
        "Grey = all other species. Blue = top-K nearest by cosine. Red star = focus. "
        "On the broken model you will see the blue cloud cluster tightly **by genus name**; "
        "on the fixed model the blue cloud disperses geographically."
    )

    st.markdown("### Top-K neighbors")
    table = cast(pd.DataFrame, neigh[["species", "cosine", "n", "km_to_focus", "same_genus"]].copy())
    table["common"] = table["species"].apply(vernacular_for)
    table["species"] = table["species"].astype(str).str.replace("_", " ")
    table["n"] = table["n"].fillna(0).astype(int)
    table = table[["species", "common", "cosine", "n", "km_to_focus", "same_genus"]]
    table.columns = pd.Index(["species", "common", "cosine", "n_occ", "km_to_focus", "same_genus"])
    st.dataframe(
        table.style.format(
            {"cosine": "{:.3f}", "km_to_focus": "{:,.0f}", "n_occ": "{:,}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("How to read this", expanded=False):
        st.markdown(
            """
- **Same-genus %**: on the broken model this often hits 70–100 %, even for
  species whose ranges do not overlap. fastText is reading the species *name*
  through character n-grams. On the fixed model the percentage drops and
  neighbors are species that actually co-occur in geohash bins.
- **Mean km to neighbors**: lower = more spatially coherent neighbors. The
  fixed model should have meaningfully lower km values for most species.
- **UMAP plot**: same species, different embedding. On the broken model the
  blue cluster sits inside a tight monogeneric cloud; on the fixed model the
  blue cluster is spatially-clustered but taxonomically mixed.
"""
        )

    st.markdown("---")
    st.markdown("### Field survey assistant — what should I look for next?")
    st.caption(
        "Real use case for the embedding. Tell it which species you have already "
        "observed at a site; it returns the species whose embedding sits closest "
        "to your observed-community centroid and that you have *not* listed yet. "
        "This is how species2vec is meant to be used downstream — as a learned "
        "co-occurrence lookup that smooths over sparse occurrence data."
    )

    seed_species = st.multiselect(
        "Species observed so far at the site",
        vocab_in_occ,
        default=[species],
        help="Pick 1–10 species. The recommender will return species likely to share the site.",
    )
    n_suggest = st.slider("How many suggestions", 5, 25, 10, key="n_suggest")

    if seed_species:
        seeds_arr = np.stack([kv[s] for s in seed_species])
        seeds_arr = seeds_arr / (np.linalg.norm(seeds_arr, axis=1, keepdims=True) + 1e-12)
        centroid = seeds_arr.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        words = list(kv.key_to_index.keys())
        M = np.stack([kv[w] for w in words])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        sims = M @ centroid

        seed_set = set(seed_species)
        order = np.argsort(-sims)
        suggestions: list[tuple[str, float]] = []
        for i in order:
            w = words[int(i)]
            if w in seed_set:
                continue
            suggestions.append((w, float(sims[int(i)])))
            if len(suggestions) >= n_suggest:
                break

        sug = pd.DataFrame(suggestions, columns=pd.Index(["species", "cosine"]))
        sug = sug.merge(cent[["species", "lat", "lon", "n"]], on="species", how="left")

        seed_genera = {s.split("_", 1)[0] for s in seed_species}
        sug["same_genus_as_seed"] = cast(pd.Series, sug["species"]).str.split("_").str[0].isin(seed_genera)
        sug["common"] = sug["species"].apply(vernacular_for)
        seed_centroid_lat = float(cent[cent["species"].isin(seed_species)]["lat"].mean())
        seed_centroid_lon = float(cent[cent["species"].isin(seed_species)]["lon"].mean())
        sug["km_to_site"] = _haversine_km(
            seed_centroid_lat, seed_centroid_lon, sug["lat"], sug["lon"]
        )
        sug["n"] = sug["n"].fillna(0).astype(int)

        novel_frac = float((~sug["same_genus_as_seed"]).mean())
        median_km = float(sug["km_to_site"].dropna().median())

        m1, m2, m3 = st.columns(3)
        m1.metric("Suggestions", len(sug))
        m2.metric(
            "Cross-genus rate",
            f"{novel_frac:.0%}",
            help="What % of suggestions are NOT congeners of your seeds. Higher = the recommender is using spatial signal, not name structure.",
        )
        m3.metric(
            "Median km to site",
            f"{median_km:,.0f} km" if not np.isnan(median_km) else "n/a",
            help="Median great-circle distance from your seed community centroid to the suggested species' range centroid.",
        )

        show = cast(pd.DataFrame, sug[["species", "common", "cosine", "n", "km_to_site", "same_genus_as_seed"]].copy())
        show["species"] = cast(pd.Series, show["species"]).astype(str).str.replace("_", " ")
        show.columns = pd.Index(["species", "common", "cosine", "n_occ", "km_to_site", "congener_of_seed"])
        st.dataframe(
            show.style.format(
                {"cosine": "{:.3f}", "km_to_site": "{:,.0f}", "n_occ": "{:,}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        sug_plot_species = seed_species + sug["species"].tolist()
        sub2 = occ[occ["species"].isin(sug_plot_species)].copy()
        sub2 = pd.concat(
            [
                g.sample(min(len(g), 200), random_state=0)
                for _, g in sub2.groupby("species", sort=False)
            ],
            ignore_index=True,
        )
        sub2["role"] = np.where(cast(pd.Series, sub2["species"]).isin(seed_species), "seed", "suggested")
        lat_c2, lon_c2, zoom2 = _auto_view(
            cast(pd.Series, sub2["decimalLatitude"]).to_numpy(),
            cast(pd.Series, sub2["decimalLongitude"]).to_numpy(),
        )
        fig_sug = px.scatter_map(
            sub2,
            lat="decimalLatitude",
            lon="decimalLongitude",
            color="role",
            hover_name="species",
            color_discrete_map={"seed": "#d62728", "suggested": "#1f77b4"},
            opacity=0.65,
            zoom=zoom2,
            center={"lat": lat_c2, "lon": lon_c2},
            map_style=base_style,
            height=500,
        )
        fig_sug.update_traces(marker=dict(size=6))
        fig_sug.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            map=dict(layers=base_layers) if base_layers else {},
        )
        st.plotly_chart(fig_sug, use_container_width=True)
        st.caption(
            "Red = seed observations. Blue = suggested species occurrences. "
            "When the blue cloud overlaps the red cloud, the embedding has "
            "successfully recovered a co-occurring community without you "
            "telling it where the site is."
        )
    else:
        st.info("Add at least one seed species above.")


if __name__ == "__main__":
    main()
