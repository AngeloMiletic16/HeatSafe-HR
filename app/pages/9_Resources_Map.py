from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.sidebar import render_app_sidebar

st.set_page_config(page_title="Cooling Centers & Resources", page_icon="🧊", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

RESOURCE_PATH = PROJECT_ROOT / "data" / "resources" / "cooling_centers.csv"

CITY_CENTER_MAP = {
    "Šibenik": {"lat": 43.7350, "lon": 15.8890, "zoom": 12},
    "Zadar": {"lat": 44.1194, "lon": 15.2314, "zoom": 12},
    "Split": {"lat": 43.5081, "lon": 16.4402, "zoom": 12},
    "Dubrovnik": {"lat": 42.6507, "lon": 18.0944, "zoom": 12},
    "Rijeka": {"lat": 45.3271, "lon": 14.4422, "zoom": 12},
    "Osijek": {"lat": 45.5550, "lon": 18.6955, "zoom": 12},
    "Zagreb": {"lat": 45.8150, "lon": 15.9819, "zoom": 12},
}

RESOURCE_COLOR_MAP = {
    "Library / public indoor space": "#2563eb",
    "Tourist info point / public indoor space": "#0891b2",
    "Hospital / health point": "#dc2626",
    "Emergency medical service / health point": "#b91c1c",
    "Park / shaded public space": "#16a34a",
    "Fountain / public outdoor point": "#06b6d4",
    "Public indoor cultural space": "#7c3aed",
    "Garden / shaded public space": "#65a30d",
    "Square / historic fountain point": "#f59e0b",
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }

    .page-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
        border-radius: 22px;
        padding: 1.35rem 1.5rem 1.2rem 1.5rem;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.16);
    }

    .page-hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .page-hero-subtitle {
        font-size: 0.98rem;
        line-height: 1.55;
        opacity: 0.95;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        min-height: 110px;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.15rem;
    }

    .metric-sub {
        font-size: 0.88rem;
        color: #64748b;
    }

    .soft-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        height: 100%;
    }

    .panel-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.7rem;
    }

    .panel-text {
        color: #334155;
        line-height: 1.65;
        font-size: 0.95rem;
    }

    .resource-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        margin-bottom: 0.85rem;
    }

    .resource-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .resource-meta {
        color: #475569;
        line-height: 1.55;
        font-size: 0.93rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .note-box {
        background: #eff6ff;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        color: #0f172a;
        margin: 0.7rem 0 1rem 0;
        line-height: 1.65;
    }

    .warning-box {
        background: #fff7ed;
        border-left: 6px solid #ea580c;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        color: #7c2d12;
        margin: 0.7rem 0 1rem 0;
        line-height: 1.65;
    }

    div.stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        padding: 0.72rem 1rem;
        font-weight: 700;
        border: none;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(29, 78, 216, 0.22);
        transition: all 0.2s ease-in-out;
    }

    div.stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(29, 78, 216, 0.30);
        background: linear-gradient(135deg, #1e293b 0%, #2563eb 100%);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">{title}</div>
            <div class="panel-text">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, color: str) -> str:
    return f'<span class="status-pill" style="background:{color};">{text}</span>'


@st.cache_data
def load_resources() -> pd.DataFrame:
    if not RESOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RESOURCE_PATH}. Save the rich CSV there first."
        )

    df = pd.read_csv(RESOURCE_PATH)

    # podrška i za stare i za nove nazive stupaca
    rename_map = {}
    if "lat" in df.columns and "latitude" not in df.columns:
        rename_map["lat"] = "latitude"
    if "lon" in df.columns and "longitude" not in df.columns:
        rename_map["lon"] = "longitude"
    df = df.rename(columns=rename_map)

    expected_cols = [
        "city", "district", "resource_name", "resource_type", "address",
        "latitude", "longitude", "hours_weekday", "hours_weekend", "phone", "website",
        "public_access", "free_access", "wheelchair_access", "water_available",
        "toilets_available", "seating_available", "shade_available", "indoor_cooling",
        "air_conditioning_confirmed", "elderly_friendly", "child_friendly",
        "medical_support_nearby", "capacity_estimate", "verified_status",
        "source_name", "source_url", "last_verified_date", "notes",
        "current_occupancy_pct", "readiness_score", "dispatch_priority"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["capacity_estimate"] = pd.to_numeric(df["capacity_estimate"], errors="coerce")
    df["current_occupancy_pct"] = pd.to_numeric(df["current_occupancy_pct"], errors="coerce")
    df["readiness_score"] = pd.to_numeric(df["readiness_score"], errors="coerce")
    df["dispatch_priority"] = pd.to_numeric(df["dispatch_priority"], errors="coerce")

    return df


def score_resource(row: pd.Series) -> float:
    score = 0.0

    verified_value = str(row.get("verified_status", "")).strip().lower()
    if verified_value == "verified":
        score += 4
    elif verified_value in ["partial", "partially verified"]:
        score += 2

    indoor_value = str(row.get("indoor_cooling", "")).strip().lower()
    if indoor_value in ["yes", "yes - indoor", "true"]:
        score += 4

    if str(row.get("air_conditioning_confirmed", "")).strip().lower() == "yes":
        score += 3
    if str(row.get("water_available", "")).strip().lower() == "yes":
        score += 2
    if str(row.get("wheelchair_access", "")).strip().lower() == "yes":
        score += 1.5
    if str(row.get("elderly_friendly", "")).strip().lower() == "yes":
        score += 1.5
    if str(row.get("child_friendly", "")).strip().lower() == "yes":
        score += 1
    if str(row.get("free_access", "")).strip().lower() == "yes":
        score += 1
    if str(row.get("public_access", "")).strip().lower() == "yes":
        score += 1
    if str(row.get("medical_support_nearby", "")).strip().lower() == "yes":
        score += 1

    readiness_score = row.get("readiness_score")
    if pd.notna(readiness_score):
        score += float(readiness_score) / 25.0

    occupancy = row.get("current_occupancy_pct")
    if pd.notna(occupancy):
        score += max(0.0, (100.0 - float(occupancy)) / 50.0)

    dispatch_priority = row.get("dispatch_priority")
    if pd.notna(dispatch_priority):
        score += float(dispatch_priority) / 5.0

    return round(score, 2)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


df = load_resources()

cities = ["Dubrovnik", "Osijek", "Rijeka", "Split", "Šibenik", "Zadar", "Zagreb"]
default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = cities.index(default_city) if default_city in cities else 0

selected_city = st.session_state.get("selected_city", default_city if default_city in cities else cities[0])

render_app_sidebar(
    selected_city=selected_city,
)

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🧊 Cooling Centers & Resource Map</div>
        <div class="page-hero-subtitle">
            Practical support layer za toplinske epizode. Ova stranica pokazuje gdje se ljudi mogu
            rashladiti, pronaći vodu, dobiti informacije ili pristupiti zdravstvenoj podršci.
            Mapa je uvijek aktivna kao operativni prikaz resursa na terenu.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2 = st.columns([1.15, 1])
with top1:
    selected_city = st.selectbox("Odaberi grad", cities, index=default_index)
    st.session_state.selected_city = selected_city

with top2:
    city_types = sorted(df[df["city"] == selected_city]["resource_type"].dropna().unique().tolist())
    selected_types = st.multiselect(
        "Filtriraj tip resursa",
        options=city_types,
        default=city_types,
    )

f1, f2, f3, f4 = st.columns(4)
with f1:
    only_indoor = st.toggle("Samo indoor cooling", value=False)
with f2:
    only_water = st.toggle("Samo voda dostupna", value=False)
with f3:
    only_accessible = st.toggle("Samo wheelchair access", value=False)
with f4:
    only_verified = st.toggle("Samo verified", value=False)

city_df = df[(df["city"] == selected_city) & (df["resource_type"].isin(selected_types))].copy()

if only_indoor:
    city_df = city_df[city_df["indoor_cooling"].astype(str).str.lower().isin(["yes", "yes - indoor", "true"])]
if only_water:
    city_df = city_df[city_df["water_available"].astype(str).str.lower() == "yes"]
if only_accessible:
    city_df = city_df[city_df["wheelchair_access"].astype(str).str.lower() == "yes"]
if only_verified:
    city_df = city_df[city_df["verified_status"].astype(str).str.lower().isin(["verified"])]

city_df["resource_score"] = city_df.apply(score_resource, axis=1)

mapped_df = city_df.dropna(subset=["latitude", "longitude"]).copy()
missing_geo_df = city_df[city_df["latitude"].isna() | city_df["longitude"].isna()].copy()

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Resources in view", str(len(city_df)), selected_city)
with m2:
    metric_card("Mapped resources", str(len(mapped_df)), "With coordinates")
with m3:
    metric_card(
        "Indoor cooling",
        str(int(city_df["indoor_cooling"].astype(str).str.lower().isin(["yes", "yes - indoor", "true"]).sum())),
        "Filtered set",
    )
with m4:
    metric_card(
        "Verified",
        str(int(city_df["verified_status"].astype(str).str.lower().isin(["verified"]).sum())),
        "Filtered set",
    )

st.markdown(
    """
    <div class="note-box">
        <b>Kako koristiti ovu stranicu:</b> kod umjerenog i višeg toplinskog rizika koristi je kao
        support layer za građane, turiste, organizatore događaja i gradske službe — ne samo za procjenu rizika,
        nego i za praktičan odgovor: gdje potražiti hlad, vodu i pomoć.
    </div>
    """,
    unsafe_allow_html=True,
)

# MAPA SE UVIJEK PRIKAZUJE
st.markdown("### Map view")

if mapped_df.empty:
    st.markdown(
        """
        <div class="warning-box">
            <b>Nema mapiranih resursa za trenutne filtere.</b><br>
            Mapa ostaje operativni fokus ove stranice, ali za odabrane filtre trenutno nema lokacija s koordinatama.
            Probaj proširiti filtre ili isključi dio ograničenja.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    center = CITY_CENTER_MAP.get(
        selected_city,
        {
            "lat": float(mapped_df["latitude"].mean()),
            "lon": float(mapped_df["longitude"].mean()),
            "zoom": 12,
        },
    )

    mapped_df["resource_type_display"] = mapped_df["resource_type"].fillna("Other")

    fig = px.scatter_mapbox(
        mapped_df,
        lat="latitude",
        lon="longitude",
        color="resource_type_display",
        hover_name="resource_name",
        hover_data={
            "district": True,
            "address": True,
            "hours_weekday": True,
            "hours_weekend": True,
            "phone": True,
            "verified_status": True,
            "resource_score": True,
            "capacity_estimate": True,
            "current_occupancy_pct": True,
            "readiness_score": True,
            "dispatch_priority": True,
            "latitude": False,
            "longitude": False,
            "resource_type_display": False,
        },
        color_discrete_map=RESOURCE_COLOR_MAP,
        zoom=center["zoom"],
        center={"lat": center["lat"], "lon": center["lon"]},
        height=680,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text="Resource type",
    )
    st.plotly_chart(fig, use_container_width=True)

info_left, info_right = st.columns(2)
with info_left:
    panel(
        "Map reading tip",
        """
        Plave i ljubičaste točke obično predstavljaju unutarnje javne prostore i informativne točke,
        crvene zdravstvenu podršku, a zelene/svjetlije vanjske resurse poput parkova i fontana.
        """,
    )
with info_right:
    panel(
        "Best operational use",
        """
        Za više razine toplinskog rizika prvo gledaj:
        <ul>
            <li>indoor cooling</li>
            <li>verified lokacije</li>
            <li>voda dostupna</li>
            <li>wheelchair-friendly pristup</li>
            <li>viši readiness score i nižu occupancy procjenu</li>
        </ul>
        """,
    )

st.divider()

st.markdown("### Top recommended resources")

top_df = city_df.sort_values(["resource_score", "verified_status"], ascending=[False, True]).head(8).copy()

if top_df.empty:
    st.warning("Nema resursa za preporuku pod ovim filterima.")
else:
    st.caption(
        "Rangiranje je interno i služi za praktičnu prioritizaciju — verified + indoor + voda + pristupačnost + readiness + raspoloživost."
    )

    for _, row in top_df.iterrows():
        badges = []

        verified_value = str(row.get("verified_status", "")).strip().lower()
        if verified_value == "verified":
            badges.append(pill("Verified", "#16a34a"))
        elif verified_value in ["partial", "partially verified"]:
            badges.append(pill("Partial", "#e67e22"))
        else:
            badges.append(pill("Needs check", "#64748b"))

        if str(row.get("indoor_cooling", "")).strip().lower() in ["yes", "yes - indoor", "true"]:
            badges.append(pill("Indoor", "#2563eb"))
        if str(row.get("water_available", "")).strip().lower() == "yes":
            badges.append(pill("Water", "#0891b2"))
        if str(row.get("wheelchair_access", "")).strip().lower() == "yes":
            badges.append(pill("Accessible", "#7c3aed"))
        if str(row.get("elderly_friendly", "")).strip().lower() == "yes":
            badges.append(pill("Elderly-friendly", "#0f766e"))
        if str(row.get("child_friendly", "")).strip().lower() == "yes":
            badges.append(pill("Child-friendly", "#db2777"))

        badge_html = "".join(badges)

        occupancy_text = (
            f"{float(row['current_occupancy_pct']):.0f}%"
            if pd.notna(row.get("current_occupancy_pct"))
            else "N/A"
        )
        readiness_text = (
            f"{float(row['readiness_score']):.0f}"
            if pd.notna(row.get("readiness_score"))
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="resource-card">
                <div class="resource-title">{row['resource_name']}</div>
                <div style="margin-bottom:0.5rem;">{badge_html}</div>
                <div class="resource-meta">
                    <b>Tip:</b> {row.get('resource_type', 'Unknown')}<br>
                    <b>Kvart/područje:</b> {row.get('district', 'Unknown')}<br>
                    <b>Adresa:</b> {row.get('address', 'Unknown')}<br>
                    <b>Radno vrijeme (tjedan):</b> {row.get('hours_weekday', 'Unknown')}<br>
                    <b>Radno vrijeme (vikend):</b> {row.get('hours_weekend', 'Unknown')}<br>
                    <b>Telefon:</b> {row.get('phone', '—') if pd.notna(row.get('phone')) else '—'}<br>
                    <b>Capacity estimate:</b> {row.get('capacity_estimate', 'N/A')}<br>
                    <b>Current occupancy:</b> {occupancy_text}<br>
                    <b>Readiness score:</b> {readiness_text}<br>
                    <b>Operational score:</b> {row.get('resource_score', 'N/A')}<br>
                    <b>Napomena:</b> {row.get('notes', '—')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

st.markdown("### Operational table")

table_cols = [
    "city", "district", "resource_name", "resource_type", "address",
    "hours_weekday", "hours_weekend", "phone", "public_access", "free_access",
    "wheelchair_access", "water_available", "toilets_available",
    "seating_available", "shade_available", "indoor_cooling",
    "air_conditioning_confirmed", "elderly_friendly", "child_friendly",
    "medical_support_nearby", "capacity_estimate", "current_occupancy_pct",
    "readiness_score", "dispatch_priority", "verified_status",
    "source_name", "source_url", "last_verified_date", "notes"
]
display_df = city_df[table_cols].copy()
st.dataframe(display_df, use_container_width=True, hide_index=True)

download_left, download_right = st.columns(2)
with download_left:
    st.download_button(
        "⬇ Download filtered resources (.csv)",
        data=csv_bytes(display_df),
        file_name=f"heatsafe_hr_resources_{selected_city}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"dl_resources_v3_{selected_city}",
    )

with download_right:
    if not missing_geo_df.empty:
        st.download_button(
            "⬇ Download resources without coordinates (.csv)",
            data=csv_bytes(missing_geo_df),
            file_name=f"heatsafe_hr_resources_missing_coordinates_{selected_city}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_resources_missing_geo_{selected_city}",
        )