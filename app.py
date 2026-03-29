from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------
# Optional OpenAI integration
# ---------------------------------------------------------------------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="UiO-66 Review Explorer",
    page_icon="🧪",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASETS = {
    "industrial_deployments": "industrial_deployments.csv",
    "benchmark_gravimetric_uptake": "benchmark_gravimetric_uptake.csv",
    "benchmark_volumetric_uptake": "benchmark_volumetric_uptake.csv",
    "mixed_gas_benchmarks": "mixed_gas_benchmarks.csv",
    "uio66_capture_design_space": "uio66_capture_design_space.csv",
    "pure_membranes": "pure_membranes.csv",
    "mixed_matrix_membranes": "mixed_matrix_membranes.csv",
    "co2_reduction_redox_potentials": "co2_reduction_redox_potentials.csv",
    "catalytic_conversion": "catalytic_conversion.csv",
    "photocatalytic_conversion": "photocatalytic_conversion.csv",
    "lca_hotspots": "lca_hotspots.csv",
    "lca_case_studies": "lca_case_studies.csv",
    "outlook_priorities": "outlook_priorities.csv",
}

PAPER_SCOPE = [
    "Performance criteria for CO2-capture MOFs",
    "Engineering UiO-66 for CO2 capture, including linker functionalisation, post-synthetic routes, hydrophobic design, pore tuning, defect chemistry, porous liquids, and composites",
    "Pure UiO-66 membranes and UiO-66-filled mixed-matrix membranes",
    "Catalytic and photocatalytic CO2 conversion",
    "Environmental impact, life-cycle assessment, shaping, regeneration, and future outlook",
]

NARRATIVE_SUMMARY = """
This explorer follows the review structure rather than the old placeholder app. It keeps the focus on benchmark capture metrics,
UiO-66 adsorption engineering, membrane and MMM performance, catalytic and photocatalytic CO2 conversion, and the paper's
life-cycle and deployment discussion.
""".strip()

SUSTAINABILITY_TAKEAWAYS = [
    "UiO-66 sustainability depends on synthesis route, shaping, regeneration duty, and service life rather than on powder uptake alone.",
    "The paper highlights large improvements for greener production routes, especially aqueous or solvent-free UiO-66-NH2 syntheses.",
    "Waste-PET linker routes are promising, but the review stresses that they still need comparative LCA rather than automatic green claims.",
    "For practical deployment, structured forms, realistic working capacity, and low sorbent loss matter more than a single headline adsorption number.",
]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data
def load_all() -> Dict[str, pd.DataFrame]:
    return {key: load_csv(filename) for key, filename in DATASETS.items()}


def normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def first_number(value: object) -> float:
    text = normalise_text(value)
    if not text or text in {"-", "–"}:
        return np.nan
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def mid_number(value: object) -> float:
    text = normalise_text(value)
    if not text or text in {"-", "–"}:
        return np.nan
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return np.nan
    vals = [float(x) for x in nums]
    return float(sum(vals) / len(vals))


def extract_temperature_k(text: object) -> float:
    txt = normalise_text(text).replace("°C", " C")
    match = re.search(r"(\d+(?:\.\d+)?)\s*K", txt)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*C", txt)
    if match:
        return float(match.group(1)) + 273.15
    return np.nan


def extract_pressure_bar(text: object) -> float:
    txt = normalise_text(text)
    if not txt:
        return np.nan
    txt = txt.replace(",", "")
    mpa_match = re.search(r"(\d+(?:\.\d+)?)\s*MPa", txt, re.IGNORECASE)
    if mpa_match:
        return float(mpa_match.group(1)) * 10.0
    atm_match = re.search(r"(\d+(?:\.\d+)?)\s*atm", txt, re.IGNORECASE)
    if atm_match:
        return float(atm_match.group(1)) * 1.01325
    bar_match = re.search(r"(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?\s*bar", txt, re.IGNORECASE)
    if bar_match:
        if bar_match.group(2):
            return (float(bar_match.group(1)) + float(bar_match.group(2))) / 2.0
        return float(bar_match.group(1))
    return np.nan


def family_from_material(name: object) -> str:
    text = normalise_text(name)
    if not text:
        return "Unknown"
    ordered = [
        "UiO-66", "UiO-67", "MOF-74", "SIFSIX", "HKUST", "ZIF", "MIL", "PCN", "UTSA", "SNU", "Cu-BTTri"
    ]
    for token in ordered:
        if token.lower() in text.lower():
            return token
    if text.startswith("NH2-UiO") or text.startswith("Opt-UiO") or "UiO" in text:
        return "UiO family"
    return "Other"


def make_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv")


def metric_row(items: List[Tuple[str, str, Optional[str]]]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, help_text = item
        col.metric(label, value, help=help_text)


def df_for_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: Optional[str] = None,
    size_default: float = 10.0,
) -> pd.DataFrame:
    out = df.copy()
    out[x] = pd.to_numeric(out[x], errors="coerce")
    out[y] = pd.to_numeric(out[y], errors="coerce")
    out = out.dropna(subset=[x, y]).copy()

    if size is not None:
        out[size] = pd.to_numeric(out[size], errors="coerce")
        median_size = out[size].median()
        if pd.isna(median_size):
            median_size = size_default
        out[size] = out[size].fillna(median_size).clip(lower=0.0)

    return out


def safe_plotly_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def safe_dataframe(df: pd.DataFrame) -> None:
    st.dataframe(df, width="stretch")


def ai_client() -> Optional["OpenAI"]:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key") or os.getenv("openai_api_key2")
    if not api_key or not OPENAI_AVAILABLE:
        return None
    return OpenAI(api_key=api_key)


def ask_ai(system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    client = ai_client()
    if client is None:
        return (
            "AI is not configured. Install the OpenAI package and set OPENAI_API_KEY as an environment variable "
            "or in Streamlit secrets."
        )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"AI error: {exc}"


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def prep_benchmark_gravimetric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sbet_num"] = out["sbet_m2_g"].map(first_number)
    out["uptake_num"] = out["co2_uptake_mmol_g"].map(first_number)
    out["qst_num"] = out["qst_kj_mol"].map(mid_number)
    out["qst_size"] = out["qst_num"]
    out["family"] = out["material"].map(family_from_material)
    return out


def prep_benchmark_volumetric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "sbet_m2_g", "crystal_density_g_cm3",
        "co2_uptake_mmol_cm3_0p1bar", "co2_uptake_mmol_cm3_0p15bar", "co2_uptake_mmol_cm3_1bar",
        "co2_uptake_mg_g_0p1bar", "co2_uptake_mg_g_0p15bar", "co2_uptake_mg_g_1bar",
    ]:
        out[col + "_num"] = out[col].map(first_number)
    out["family"] = out["adsorbent"].map(family_from_material)
    return out


def prep_mixed_gas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "post_combustion_co2_uptake_mmol_g",
        "post_combustion_co2_n2_selectivity",
        "pre_combustion_co2_uptake_mmol_g",
        "pre_combustion_co2_h2_selectivity",
        "qst_kj_mol",
    ]:
        out[col + "_num"] = out[col].map(mid_number)
    out["qst_size"] = out["qst_kj_mol_num"]
    out["family"] = out["adsorbent"].map(family_from_material)
    return out


def prep_capture_design(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "sbet_m2_g", "co2_uptake_mmol_g", "n2_uptake_mmol_g", "ch4_uptake_mmol_g", "h2_uptake_mmol_g",
        "selectivity_co2_n2", "selectivity_co2_ch4", "selectivity_co2_h2",
    ]:
        out[col + "_num"] = out[col].map(mid_number)
    out["temperature_k"] = out["conditions"].map(extract_temperature_k)
    out["pressure_bar"] = out["conditions"].map(extract_pressure_bar)
    out["family"] = out["material"].map(family_from_material)
    return out


def prep_pure_membranes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gas_i_num"] = out["gas_i_permeance_1e8_mol_m2_s_pa"].map(mid_number)
    out["gas_j_num"] = out["gas_j_permeance_1e8_mol_m2_s_pa"].map(mid_number)
    out["separation_num"] = out["separation_factor"].map(mid_number)
    out["temperature_k"] = out["conditions"].map(extract_temperature_k)
    out["pressure_bar"] = out["conditions"].map(extract_pressure_bar)
    return out


def prep_mmm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "loading_pct", "co2_permeability_or_gpu", "n2_permeability_or_gpu", "ch4_permeability_or_gpu",
        "h2_permeability_or_gpu", "selectivity_co2_n2", "selectivity_co2_ch4", "selectivity_co2_h2",
    ]:
        out[col + "_num"] = out[col].map(mid_number)
    out["loading_size"] = out["loading_pct_num"]
    out["temperature_k"] = out["conditions"].map(extract_temperature_k)
    out["pressure_bar"] = out["conditions"].map(extract_pressure_bar)
    return out


def prep_redox(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["potential_num"] = out["standard_potential_v_vs_she"].map(first_number)
    return out


def prep_catalysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["yield_num"] = out["yield_pct"].map(mid_number)
    out["selectivity_num"] = out["selectivity_pct"].map(mid_number)
    out["temperature_k"] = out["reaction_conditions"].map(extract_temperature_k)
    out["pressure_bar"] = out["reaction_conditions"].map(extract_pressure_bar)
    out["product_group"] = out["product"].fillna("").replace("", "Unspecified")
    out["co_catalyst_clean"] = out["co_catalyst"].fillna("").replace("", "Unspecified")
    return out


def prep_photocatalysis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["efficiency_num"] = out["photocatalytic_efficiency_umol_g_h"].map(mid_number)
    out["product_group"] = out["product"].fillna("").replace("", "Unspecified")
    out["sacrificial_agent_clean"] = out["sacrificial_agent"].fillna("").replace("", "Unspecified")
    out["irradiation_clean"] = out["irradiation"].fillna("").replace("", "Unspecified")
    return out


# ---------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------
def render_home(dfs: Dict[str, pd.DataFrame]) -> None:
    st.title("UiO-66 Review Explorer")
    st.caption(
        "Interactive companion app for the review on UiO-66-derived MOFs for CO2 capture, separation, and conversion."
    )
    st.write(NARRATIVE_SUMMARY)

    capture_df = prep_capture_design(dfs["uio66_capture_design_space"])
    mmm_df = prep_mmm(dfs["mixed_matrix_membranes"])
    pure_mem_df = prep_pure_membranes(dfs["pure_membranes"])
    cat_df = prep_catalysis(dfs["catalytic_conversion"])
    photo_df = prep_photocatalysis(dfs["photocatalytic_conversion"])

    metric_row([
        ("Industrial examples", f"{len(dfs['industrial_deployments']):,}", "Industrial MOF-based capture examples cited in the review"),
        ("UiO-66 capture records", f"{len(capture_df):,}", "UiO-66 adsorption rows extracted from the review"),
        ("Membrane records", f"{len(mmm_df) + len(pure_mem_df):,}", "Pure membrane and MMM rows"),
        ("Catalytic records", f"{len(cat_df):,}", "Catalytic CO2 conversion rows"),
        ("Photocatalytic records", f"{len(photo_df):,}", "Photocatalytic CO2 conversion rows"),
    ])

    col1, col2 = st.columns([1.1, 1.4])

    with col1:
        st.markdown("#### Scope covered in the app")
        for item in PAPER_SCOPE:
            st.markdown(f"- {item}")

        st.markdown("#### Sustainability signals highlighted by the paper")
        for item in SUSTAINABILITY_TAKEAWAYS:
            st.markdown(f"- {item}")

    with col2:
        industrial = dfs["industrial_deployments"].copy()
        industrial["capacity_num"] = industrial["capacity_tco2_day"].map(mid_number).fillna(0.0)
        fig = px.bar(
            industrial,
            x="company",
            y="capacity_num",
            color="deployment_stage",
            text="capacity_tco2_day",
            title="Industrial deployment examples cited in the review",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Approximate capacity scale (t CO2/day)")
        safe_plotly_chart(fig)

    st.divider()

    redox = prep_redox(dfs["co2_reduction_redox_potentials"]).dropna(subset=["potential_num"]).copy()
    if not redox.empty:
        fig2 = px.bar(
            redox.sort_values("potential_num", ascending=False),
            x="reaction",
            y="potential_num",
            title="Reference electrochemical reduction potentials listed in the review",
        )
        fig2.update_layout(xaxis_title="", yaxis_title="E° vs SHE (V)")
        fig2.update_xaxes(tickangle=-25)
        safe_plotly_chart(fig2)


def render_benchmarks(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Capture benchmarks")

    view = st.radio(
        "Choose benchmark dataset",
        ["Gravimetric uptake at 298 K and 1 bar", "Volumetric uptake at 298 K", "Mixed-gas benchmarks"],
        horizontal=True,
    )

    if view == "Gravimetric uptake at 298 K and 1 bar":
        df = prep_benchmark_gravimetric(dfs["benchmark_gravimetric_uptake"])
        families = sorted(df["family"].dropna().unique().tolist())
        default_families = [x for x in ["UiO-66", "MOF-74", "Other"] if x in families]
        selected = st.multiselect("Family", families, default=default_families or families[:5])
        if selected:
            df = df[df["family"].isin(selected)]

        plot_df = df_for_scatter(df, x="sbet_num", y="uptake_num", size="qst_size", size_default=25.0)
        fig = px.scatter(
            plot_df,
            x="sbet_num",
            y="uptake_num",
            color="family",
            size="qst_size",
            hover_data=["material", "co2_uptake_mmol_g", "qst_kj_mol", "reference"],
            title="Surface area versus CO2 uptake",
        )
        fig.update_layout(xaxis_title="SBET (m²/g)", yaxis_title="CO2 uptake (mmol/g)")
        safe_plotly_chart(fig)

        top = df.dropna(subset=["uptake_num"]).sort_values("uptake_num", ascending=False).head(20)
        safe_dataframe(top[["material", "family", "sbet_m2_g", "co2_uptake_mmol_g", "qst_kj_mol", "reference"]])
        make_download_button(df, "Download filtered benchmark table", "benchmark_gravimetric_filtered.csv")

    elif view == "Volumetric uptake at 298 K":
        df = prep_benchmark_volumetric(dfs["benchmark_volumetric_uptake"])
        pressure_choice = st.selectbox("Pressure", ["0.1 bar", "0.15 bar", "1 bar"], index=2)
        mapping = {
            "0.1 bar": ("co2_uptake_mmol_cm3_0p1bar_num", "co2_uptake_mg_g_0p1bar_num", "co2_uptake_mmol_cm3_0p1bar"),
            "0.15 bar": ("co2_uptake_mmol_cm3_0p15bar_num", "co2_uptake_mg_g_0p15bar_num", "co2_uptake_mmol_cm3_0p15bar"),
            "1 bar": ("co2_uptake_mmol_cm3_1bar_num", "co2_uptake_mg_g_1bar_num", "co2_uptake_mmol_cm3_1bar"),
        }
        vol_col, _, raw_col = mapping[pressure_choice]

        plot_df = df_for_scatter(df, x="crystal_density_g_cm3_num", y=vol_col, size="sbet_m2_g_num", size_default=500.0)
        fig = px.scatter(
            plot_df,
            x="crystal_density_g_cm3_num",
            y=vol_col,
            color="family",
            size="sbet_m2_g_num",
            hover_data=["adsorbent", raw_col, "co2_uptake_mg_g_1bar", "reference"],
            title=f"Density versus volumetric CO2 uptake at {pressure_choice}",
        )
        fig.update_layout(xaxis_title="Crystal density (g/cm³)", yaxis_title=f"Volumetric CO2 uptake at {pressure_choice}")
        safe_plotly_chart(fig)

        top = df.dropna(subset=[vol_col]).sort_values(vol_col, ascending=False).head(15)
        safe_dataframe(top[["adsorbent", "crystal_density_g_cm3", raw_col, "co2_uptake_mg_g_1bar", "reference"]])
        make_download_button(df, "Download volumetric benchmark table", "benchmark_volumetric_filtered.csv")

    else:
        df = prep_mixed_gas(dfs["mixed_gas_benchmarks"])
        mode = st.radio("Scenario", ["Post-combustion CO2/N2", "Pre-combustion CO2/H2"], horizontal=True)
        if mode == "Post-combustion CO2/N2":
            xcol = "post_combustion_co2_uptake_mmol_g_num"
            ycol = "post_combustion_co2_n2_selectivity_num"
            raw_cols = ["post_combustion_co2_uptake_mmol_g", "post_combustion_co2_n2_selectivity"]
            title = "Post-combustion mixed-gas benchmarks"
        else:
            xcol = "pre_combustion_co2_uptake_mmol_g_num"
            ycol = "pre_combustion_co2_h2_selectivity_num"
            raw_cols = ["pre_combustion_co2_uptake_mmol_g", "pre_combustion_co2_h2_selectivity"]
            title = "Pre-combustion mixed-gas benchmarks"

        filtered = df_for_scatter(df, x=xcol, y=ycol, size="qst_size", size_default=30.0)
        fig = px.scatter(
            filtered,
            x=xcol,
            y=ycol,
            color="family",
            size="qst_size",
            hover_data=["adsorbent", raw_cols[0], raw_cols[1], "qst_kj_mol", "reference"],
            title=title,
            log_y=True,
        )
        fig.update_layout(xaxis_title="CO2 uptake (mmol/g)", yaxis_title="Selectivity")
        safe_plotly_chart(fig)

        top = filtered.sort_values(ycol, ascending=False).head(20)
        safe_dataframe(top[["adsorbent", raw_cols[0], raw_cols[1], "qst_kj_mol", "reference"]])
        make_download_button(filtered, "Download mixed-gas table", "mixed_gas_filtered.csv")


def render_design_space(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("UiO-66 adsorption design space")

    df = prep_capture_design(dfs["uio66_capture_design_space"])
    categories = sorted(df["category"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Strategy category",
        categories,
        default=categories[:4] if len(categories) > 4 else categories,
    )

    metric_mode = st.radio(
        "Explore by",
        ["Pure CO2 uptake", "CO2/N2 selectivity", "CO2/CH4 selectivity", "CO2/H2 selectivity"],
        horizontal=True,
    )

    filtered = df.copy()
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    temp_range = st.slider("Temperature range (K)", 250, 500, (280, 320))
    filtered = filtered[
        filtered["temperature_k"].isna()
        | ((filtered["temperature_k"] >= temp_range[0]) & (filtered["temperature_k"] <= temp_range[1]))
    ]

    if metric_mode == "Pure CO2 uptake":
        xcol, ycol = "sbet_m2_g_num", "co2_uptake_mmol_g_num"
        ylab = "CO2 uptake (mmol/g)"
        hover_cols = ["material", "conditions", "co2_uptake_mmol_g", "reference"]
    elif metric_mode == "CO2/N2 selectivity":
        xcol, ycol = "co2_uptake_mmol_g_num", "selectivity_co2_n2_num"
        ylab = "CO2/N2 selectivity"
        hover_cols = ["material", "conditions", "co2_uptake_mmol_g", "selectivity_co2_n2", "reference"]
    elif metric_mode == "CO2/CH4 selectivity":
        xcol, ycol = "co2_uptake_mmol_g_num", "selectivity_co2_ch4_num"
        ylab = "CO2/CH4 selectivity"
        hover_cols = ["material", "conditions", "co2_uptake_mmol_g", "selectivity_co2_ch4", "reference"]
    else:
        xcol, ycol = "co2_uptake_mmol_g_num", "selectivity_co2_h2_num"
        ylab = "CO2/H2 selectivity"
        hover_cols = ["material", "conditions", "co2_uptake_mmol_g", "selectivity_co2_h2", "reference"]

    filtered = df_for_scatter(filtered, x=xcol, y=ycol)
    fig = px.scatter(
        filtered,
        x=xcol,
        y=ycol,
        color="category",
        symbol="family",
        hover_data=hover_cols,
        title=f"UiO-66 design space: {metric_mode}",
    )
    if "selectivity" in metric_mode:
        fig.update_yaxes(type="log")
        fig.update_layout(xaxis_title="CO2 uptake (mmol/g)", yaxis_title=ylab)
    else:
        fig.update_layout(xaxis_title="SBET (m²/g)", yaxis_title=ylab)
    safe_plotly_chart(fig)

    category_summary = (
        filtered.groupby("category", dropna=False)
        .agg(
            records=("material", "count"),
            median_co2_uptake=("co2_uptake_mmol_g_num", "median"),
            median_co2_n2_selectivity=("selectivity_co2_n2_num", "median"),
            median_temperature_k=("temperature_k", "median"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        st.markdown("#### Category summary")
        safe_dataframe(category_summary)
    with col2:
        fig2 = px.bar(
            category_summary,
            x="category",
            y="records",
            title="How many records are available per strategy category",
            text="records",
        )
        fig2.update_xaxes(tickangle=-30)
        safe_plotly_chart(fig2)

    ranking_mode = st.selectbox(
        "Ranking table",
        ["Highest CO2 uptake", "Highest CO2/N2 selectivity", "Highest CO2/CH4 selectivity", "Highest CO2/H2 selectivity"],
    )
    ranking_map = {
        "Highest CO2 uptake": "co2_uptake_mmol_g_num",
        "Highest CO2/N2 selectivity": "selectivity_co2_n2_num",
        "Highest CO2/CH4 selectivity": "selectivity_co2_ch4_num",
        "Highest CO2/H2 selectivity": "selectivity_co2_h2_num",
    }
    rank_col = ranking_map[ranking_mode]
    top = filtered.dropna(subset=[rank_col]).sort_values(rank_col, ascending=False).head(25)
    st.markdown("#### Top rows in the current view")
    safe_dataframe(
        top[[
            "material", "category", "conditions", "co2_uptake_mmol_g", "selectivity_co2_n2",
            "selectivity_co2_ch4", "selectivity_co2_h2", "reference"
        ]]
    )
    make_download_button(filtered, "Download filtered UiO-66 design-space table", "uio66_design_space_filtered.csv")


def render_membranes(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Membranes and mixed-matrix membranes")

    subtab1, subtab2 = st.tabs(["Pure UiO-66-based membranes", "UiO-66-filled MMMs"])

    with subtab1:
        df = prep_pure_membranes(dfs["pure_membranes"])
        pairs = sorted(df["binary_gas_pair"].dropna().unique().tolist())
        pair = st.selectbox("Binary gas pair", pairs, index=pairs.index("CO2/N2") if "CO2/N2" in pairs else 0)

        filtered = df[df["binary_gas_pair"] == pair].copy()
        filtered = df_for_scatter(filtered, x="gas_i_num", y="separation_num")

        fig = px.scatter(
            filtered,
            x="gas_i_num",
            y="separation_num",
            color="membrane",
            hover_data=["conditions", "gas_i_permeance_1e8_mol_m2_s_pa", "gas_j_permeance_1e8_mol_m2_s_pa", "reference"],
            title=f"Pure membrane performance for {pair}",
        )
        fig.update_layout(xaxis_title="Gas i permeance", yaxis_title="Separation factor")
        fig.update_yaxes(type="log")
        safe_plotly_chart(fig)

        safe_dataframe(filtered[[
            "membrane", "conditions", "binary_gas_pair",
            "gas_i_permeance_1e8_mol_m2_s_pa", "gas_j_permeance_1e8_mol_m2_s_pa",
            "separation_factor", "reference"
        ]])
        make_download_button(filtered, "Download filtered pure membrane table", "pure_membranes_filtered.csv")

    with subtab2:
        df = prep_mmm(dfs["mixed_matrix_membranes"])
        mode = st.radio("MMM metric", ["CO2/N2", "CO2/CH4", "CO2/H2"], horizontal=True)
        if mode == "CO2/N2":
            sel_col = "selectivity_co2_n2_num"
            sel_raw = "selectivity_co2_n2"
        elif mode == "CO2/CH4":
            sel_col = "selectivity_co2_ch4_num"
            sel_raw = "selectivity_co2_ch4"
        else:
            sel_col = "selectivity_co2_h2_num"
            sel_raw = "selectivity_co2_h2"

        categories = sorted(df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "MMM categories",
            categories,
            default=categories[:4] if len(categories) > 4 else categories,
        )

        filtered = df.copy()
        if selected_categories:
            filtered = filtered[filtered["category"].isin(selected_categories)]

        loading_max = st.slider("Maximum filler loading (%)", 0.0, 100.0, 40.0, 1.0)
        filtered = filtered[(filtered["loading_pct_num"].isna()) | (filtered["loading_pct_num"] <= loading_max)]
        filtered = df_for_scatter(filtered, x="co2_permeability_or_gpu_num", y=sel_col, size="loading_size", size_default=10.0)

        fig = px.scatter(
            filtered,
            x="co2_permeability_or_gpu_num",
            y=sel_col,
            color="category",
            size="loading_size",
            hover_data=["polymer", "filler", "loading_pct", "conditions", sel_raw, "reference"],
            title=f"MMM trade-off map for {mode}",
        )
        fig.update_xaxes(type="log", title="CO2 permeability / permeance")
        fig.update_yaxes(type="log", title=f"{mode} selectivity")
        safe_plotly_chart(fig)

        top = filtered.sort_values(sel_col, ascending=False).head(30)
        safe_dataframe(top[[
            "polymer", "filler", "category", "loading_pct", "conditions",
            "co2_permeability_or_gpu", sel_raw, "reference"
        ]])

        polymer_counts = filtered["polymer"].value_counts().head(15).reset_index()
        polymer_counts.columns = ["polymer", "records"]
        fig2 = px.bar(polymer_counts, x="polymer", y="records", title="Most common polymer matrices in the current MMM view")
        fig2.update_xaxes(tickangle=-30)
        safe_plotly_chart(fig2)
        make_download_button(filtered, "Download filtered MMM table", "mixed_matrix_membranes_filtered.csv")


def render_catalysis(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Catalytic CO2 conversion")

    df = prep_catalysis(dfs["catalytic_conversion"])
    products = sorted(df["product_group"].dropna().unique().tolist())
    selected_products = st.multiselect("Products", products, default=products[:5] if len(products) > 5 else products)

    filtered = df.copy()
    if selected_products:
        filtered = filtered[filtered["product_group"].isin(selected_products)]

    mode = st.radio("Metric", ["Yield", "Selectivity"], horizontal=True)
    metric_col = "yield_num" if mode == "Yield" else "selectivity_num"
    raw_col = "yield_pct" if mode == "Yield" else "selectivity_pct"

    chart_df = filtered.dropna(subset=[metric_col]).copy()
    fig = px.bar(
        chart_df.sort_values(metric_col, ascending=False).head(40),
        x="catalyst",
        y=metric_col,
        color="product_group",
        hover_data=["co_catalyst_clean", "reaction_conditions", raw_col, "reference"],
        title=f"Top catalytic systems by {mode.lower()}",
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(yaxis_title=f"{mode} (%)")
    safe_plotly_chart(fig)

    col1, col2 = st.columns([1.25, 1.0])
    with col1:
        safe_dataframe(filtered[[
            "catalyst", "co_catalyst_clean", "reaction_conditions", "product_group", "yield_pct", "selectivity_pct", "reference"
        ]])
    with col2:
        cocat_counts = filtered["co_catalyst_clean"].value_counts().head(12).reset_index()
        cocat_counts.columns = ["co_catalyst", "records"]
        fig2 = px.pie(cocat_counts, values="records", names="co_catalyst", title="Co-catalyst distribution in the current view")
        safe_plotly_chart(fig2)

    make_download_button(filtered, "Download filtered catalytic table", "catalytic_conversion_filtered.csv")


def render_photocatalysis(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Photocatalytic CO2 conversion")

    df = prep_photocatalysis(dfs["photocatalytic_conversion"])
    products = sorted(df["product_group"].dropna().unique().tolist())
    selected_products = st.multiselect(
        "Photocatalytic products",
        products,
        default=products[:4] if len(products) > 4 else products,
    )

    filtered = df.copy()
    if selected_products:
        filtered = filtered[filtered["product_group"].isin(selected_products)]

    filtered = filtered.dropna(subset=["efficiency_num"]).copy()

    fig = px.bar(
        filtered.sort_values("efficiency_num", ascending=False).head(40),
        x="photocatalyst",
        y="efficiency_num",
        color="product_group",
        hover_data=["sacrificial_agent_clean", "irradiation_clean", "reference"],
        title="Top photocatalytic efficiencies reported in the review",
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(yaxis_title="Photocatalytic efficiency (µmol/g h)")
    safe_plotly_chart(fig)

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        safe_dataframe(filtered[[
            "photocatalyst", "sacrificial_agent_clean", "irradiation_clean", "product_group",
            "photocatalytic_efficiency_umol_g_h", "reference"
        ]])
    with col2:
        irr = filtered["irradiation_clean"].value_counts().reset_index()
        irr.columns = ["irradiation", "records"]
        fig2 = px.bar(irr, x="irradiation", y="records", title="Irradiation modes in the current view")
        safe_plotly_chart(fig2)

    make_download_button(filtered, "Download filtered photocatalytic table", "photocatalytic_conversion_filtered.csv")


def render_sustainability(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Sustainability, LCA, and outlook")

    case_df = dfs["lca_case_studies"].copy()
    hotspot_df = dfs["lca_hotspots"].copy()
    outlook_df = dfs["outlook_priorities"].copy()

    metric_row([
        ("LCA case studies", f"{len(case_df):,}", "Narrative case studies extracted from the review"),
        ("Hotspot categories", f"{len(hotspot_df):,}", "Main life-cycle hotspots table"),
        ("Outlook priorities", f"{len(outlook_df):,}", "Challenges and future work structured from the review"),
    ])

    st.markdown("#### Case-study view")
    case_choice = st.selectbox("Case study", case_df["case_study"].tolist())
    row = case_df[case_df["case_study"] == case_choice].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Application", row["application"])
    c2.metric("Environmental improvement (%)", normalise_text(row["environmental_burden_reduction_pct"]) or "Not stated")
    c3.metric("Cost improvement (%)", normalise_text(row["cost_reduction_pct"]) or "Not stated")

    st.info(f"**Route or system:** {row['route_or_system']}")
    if normalise_text(row["reported_gwp_kgco2e_per_kg"]):
        st.write(f"**Reported GWP:** {row['reported_gwp_kgco2e_per_kg']} kg CO2-eq per kg")
    if normalise_text(row["reported_cost_usd_per_kg"]):
        st.write(f"**Reported production cost:** {row['reported_cost_usd_per_kg']} USD per kg")
    st.write(f"**Key metric:** {row['other_key_metric']}")
    st.write(f"**Why it matters:** {row['notes']}")

    case_plot = case_df.copy()
    case_plot["env_reduction_num"] = case_plot["environmental_burden_reduction_pct"].map(first_number)
    case_plot["cost_reduction_num"] = case_plot["cost_reduction_pct"].map(first_number)
    fig = px.bar(
        case_plot,
        x="case_study",
        y=["env_reduction_num", "cost_reduction_num"],
        barmode="group",
        title="Reported sustainability improvements from case studies",
    )
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(yaxis_title="Improvement (%)", legend_title="")
    safe_plotly_chart(fig)

    st.markdown("#### Main life-cycle hotspots")
    safe_dataframe(hotspot_df)

    st.markdown("#### Outlook priorities")
    area = st.selectbox("Challenge area", sorted(outlook_df["challenge_area"].unique().tolist()))
    filtered = outlook_df[outlook_df["challenge_area"] == area]
    safe_dataframe(filtered)

    fig2 = px.bar(
        outlook_df.groupby("challenge_area").size().reset_index(name="count"),
        x="challenge_area",
        y="count",
        title="How the review's future priorities are distributed",
    )
    safe_plotly_chart(fig2)

    make_download_button(case_df, "Download LCA case studies", "lca_case_studies.csv")
    make_download_button(hotspot_df, "Download LCA hotspots", "lca_hotspots.csv")
    make_download_button(outlook_df, "Download outlook priorities", "outlook_priorities.csv")


def render_data_browser(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Data browser")

    dataset_name = st.selectbox("Dataset", list(dfs.keys()))
    df = dfs[dataset_name]
    st.write(f"Rows: {len(df):,}")
    safe_dataframe(df)
    make_download_button(df, "Download this dataset", f"{dataset_name}.csv")


def build_context_snippets(dfs: Dict[str, pd.DataFrame], focus: str) -> str:
    capture_df = prep_capture_design(dfs["uio66_capture_design_space"])
    mmm_df = prep_mmm(dfs["mixed_matrix_membranes"])
    cat_df = prep_catalysis(dfs["catalytic_conversion"])
    photo_df = prep_photocatalysis(dfs["photocatalytic_conversion"])
    case_df = dfs["lca_case_studies"]

    snippets = [
        "You are answering questions about a cleaned review dataset on UiO-66-derived MOFs for CO2 capture, separation, and conversion.",
        "Use only the provided dataset context. If something is missing or ambiguous, say so clearly.",
    ]

    if focus == "Adsorption":
        top_uptake = capture_df.dropna(subset=["co2_uptake_mmol_g_num"]).sort_values("co2_uptake_mmol_g_num", ascending=False).head(20)
        top_sel = capture_df.dropna(subset=["selectivity_co2_n2_num"]).sort_values("selectivity_co2_n2_num", ascending=False).head(20)
        snippets.append("Top UiO-66 adsorption rows by CO2 uptake:\n" + top_uptake[["material", "category", "conditions", "co2_uptake_mmol_g", "selectivity_co2_n2", "reference"]].to_csv(index=False))
        snippets.append("Top UiO-66 adsorption rows by CO2/N2 selectivity:\n" + top_sel[["material", "category", "conditions", "co2_uptake_mmol_g", "selectivity_co2_n2", "reference"]].to_csv(index=False))
    elif focus == "Membranes":
        top_mmm = mmm_df.dropna(subset=["co2_permeability_or_gpu_num"]).sort_values("co2_permeability_or_gpu_num", ascending=False).head(25)
        snippets.append("Representative MMM rows:\n" + top_mmm[["polymer", "filler", "category", "loading_pct", "conditions", "co2_permeability_or_gpu", "selectivity_co2_n2", "selectivity_co2_ch4", "reference"]].to_csv(index=False))
    elif focus == "Catalysis":
        top_cat = cat_df.dropna(subset=["yield_num"]).sort_values("yield_num", ascending=False).head(25)
        snippets.append("Representative catalytic rows:\n" + top_cat[["catalyst", "co_catalyst_clean", "reaction_conditions", "product_group", "yield_pct", "selectivity_pct", "reference"]].to_csv(index=False))
    elif focus == "Photocatalysis":
        top_photo = photo_df.dropna(subset=["efficiency_num"]).sort_values("efficiency_num", ascending=False).head(25)
        snippets.append("Representative photocatalytic rows:\n" + top_photo[["photocatalyst", "sacrificial_agent_clean", "irradiation_clean", "product_group", "photocatalytic_efficiency_umol_g_h", "reference"]].to_csv(index=False))
    else:
        snippets.append("Sustainability case studies:\n" + case_df.to_csv(index=False))
        snippets.append("LCA hotspots:\n" + dfs["lca_hotspots"].to_csv(index=False))

    return "\n\n".join(snippets)


def render_ai_assistant(dfs: Dict[str, pd.DataFrame]) -> None:
    st.header("Ask the review")
    st.caption("Optional AI assistant integrated into the same Python file. It is restricted to the loaded dataset context.")

    focus = st.selectbox("Context focus", ["Adsorption", "Membranes", "Catalysis", "Photocatalysis", "Sustainability"])
    question = st.text_area(
        "Question",
        value="What are the most promising UiO-66 directions for practical deployment, and what are the trade-offs?",
        height=140,
    )

    with st.expander("AI settings"):
        model = st.text_input("Model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.slider("Max tokens", 200, 1200, 650, 50)

    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        system_prompt = build_context_snippets(dfs, focus)
        with st.spinner("Querying AI..."):
            answer = ask_ai(system_prompt, question, model=model, temperature=temperature, max_tokens=max_tokens)
        st.markdown("#### Response")
        st.info(answer)

    if ai_client() is None:
        st.warning("AI is currently unavailable because no API key is configured or the OpenAI package is missing.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dfs = load_all()

    missing = [name for name, df in dfs.items() if df.empty]
    if missing:
        st.error("Some expected datasets are missing or empty: " + ", ".join(missing))

    tab_home, tab_bench, tab_design, tab_mem, tab_cat, tab_photo, tab_sus, tab_data, tab_ai = st.tabs([
        "Home",
        "Benchmarks",
        "UiO-66 design space",
        "Membranes",
        "Catalysis",
        "Photocatalysis",
        "Sustainability",
        "Data browser",
        "Ask the review",
    ])

    with tab_home:
        render_home(dfs)
    with tab_bench:
        render_benchmarks(dfs)
    with tab_design:
        render_design_space(dfs)
    with tab_mem:
        render_membranes(dfs)
    with tab_cat:
        render_catalysis(dfs)
    with tab_photo:
        render_photocatalysis(dfs)
    with tab_sus:
        render_sustainability(dfs)
    with tab_data:
        render_data_browser(dfs)
    with tab_ai:
        render_ai_assistant(dfs)


if __name__ == "__main__":
    main()
