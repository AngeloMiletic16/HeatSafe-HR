from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary, readiness_to_color, risk_to_color
from src.forecast_engine import make_ml_forecast

st.set_page_config(page_title="Command Dashboard", page_icon="🛰️", layout="wide")

CITY_IMAGE_DIR = PROJECT_ROOT / "app" / "assets" / "cities"

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

READINESS_ORDER = {
    "Critical Preparedness": 4,
    "Elevated Readiness": 3,
    "Prepared": 2,
    "Monitoring": 1,
}

RISK_ORDER = {
    "Vrlo visok": 4,
    "Visok": 3,
    "Umjeren": 2,
    "Nizak": 1,
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
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

    .alert-box {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border: 1px solid rgba(234, 88, 12, 0.18);
        border-left: 8px solid #ea580c;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.05);
    }

    .alert-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #9a3412;
        margin-bottom: 0.3rem;
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
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 0.15rem;
    }

    .metric-sub {
        font-size: 0.88rem;
        color: #64748b;
    }

    .city-card {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        margin-bottom: 1rem;
    }

    .city-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .mini-muted {
        font-size: 0.88rem;
        color: #64748b;
    }

    .section-box {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.42rem 0.82rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.94rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def slugify_city_name(city_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", city_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_")


def find_city_image(city_name: str) -> Path | None:
    slug = slugify_city_name(city_name)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = CITY_IMAGE_DIR / f"{slug}{ext}"
        if candidate.exists():
            return candidate
    return None


def badge(text: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{text}</span>',
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


@st.cache_data(ttl=1800)
def load_city_forecast(
    city_name: str,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> pd.DataFrame:
    return make_ml_forecast(
        city_name,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )


def compute_priority_score(summary: dict) -> float:
    score = 0.0
    score += summary["next_7d_peak_score"] * 1.0
    score += summary["next_24h_score"] * 0.6
    score += summary["high_risk_days"] * 8.0
    score += READINESS_ORDER.get(summary["readiness_status"], 1) * 10.0
    score += RISK_ORDER.get(summary["next_7d_peak_level"], 1) * 4.0
    return round(score, 2)


def build_command_dataset(
    cities: list[str],
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    forecast_map = {}

    for city in cities:
        forecast_df = load_city_forecast(
            city_name=city,
            temperature_delta=temperature_delta,
            humidity_delta=humidity_delta,
            wind_delta=wind_delta,
        )
        forecast_map[city] = forecast_df

        summary = build_city_readiness_summary(city, forecast_df)
        next24_row = forecast_df.sort_values("date").iloc[0]
        peak_row = forecast_df.sort_values(
            ["heuristic_risk_score", "apparent_temp_max", "ml_prediction_confidence"],
            ascending=[False, False, False],
        ).iloc[0]

        rows.append(
            {
                "city": city,
                "readiness_status": summary["readiness_status"],
                "next_24h_level": summary["next_24h_level"],
                "next_24h_score": summary["next_24h_score"],
                "next_24h_ml_label": summary["next_24h_ml_label"],
                "next_24h_confidence": summary["next_24h_confidence"],
                "next_72h_peak_level": summary["next_72h_peak_level"],
                "next_72h_peak_score": summary["next_72h_peak_score"],
                "next_7d_peak_level": summary["next_7d_peak_level"],
                "next_7d_peak_score": summary["next_7d_peak_score"],
                "next_7d_peak_date": summary["next_7d_peak_date"],
                "high_risk_days": summary["high_risk_days"],
                "peak_temp_max": peak_row["temp_max"],
                "peak_apparent_temp_max": peak_row["apparent_temp_max"],
                "peak_ml_label": peak_row["ml_predicted_label"],
                "peak_ml_confidence": peak_row["ml_prediction_confidence"],
                "priority_score": compute_priority_score(summary),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["priority_score", "next_7d_peak_score", "high_risk_days"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df, forecast_map


st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🛰️ Multi-City Command Dashboard</div>
        <div class="page-hero-subtitle">
            Nacionalni pregled toplinskog rizika za hrvatske gradove. Ova stranica rangira gradove,
            prikazuje readiness status, next 7d peak, high-risk days i odmah pokazuje tko zahtijeva
            najviše pažnje.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

scenario_enabled = st.toggle("Scenario mode za sve gradove", value=True)

if scenario_enabled:
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature_delta = st.slider("Promjena temperature (°C)", -2, 12, 6, 1)
    with c2:
        humidity_delta = st.slider("Promjena vlage (%)", -20, 30, 10, 1)
    with c3:
        wind_delta = st.slider("Promjena vjetra (m/s)", -8, 5, -3, 1)
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

try:
    command_df, forecast_map = build_command_dataset(
        CITIES,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
except Exception as exc:
    st.error(f"Command dashboard nije dostupan: {exc}")
    st.stop()

top_city = command_df.iloc[0]

st.markdown(
    f"""
    <div class="alert-box">
        <div class="alert-title">Top alert city: {top_city['city']}</div>
        <div>
            Readiness status: <b>{top_city['readiness_status']}</b> |
            Next 7d peak: <b>{top_city['next_7d_peak_level']}</b> |
            Peak score: <b>{top_city['next_7d_peak_score']:.1f}</b> |
            Peak date: <b>{pd.to_datetime(top_city['next_7d_peak_date']).strftime('%d.%m.%Y.')}</b> |
            High-risk days: <b>{int(top_city['high_risk_days'])}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Top city", top_city["city"], top_city["readiness_status"])
with m2:
    metric_card("Highest peak score", f"{top_city['next_7d_peak_score']:.1f}", top_city["next_7d_peak_level"])
with m3:
    metric_card("Most urgent date", pd.to_datetime(top_city["next_7d_peak_date"]).strftime("%d.%m.%Y."), "Peak day")
with m4:
    metric_card("Cities in system", str(len(command_df)), "Croatia command view")

st.markdown("## Who needs the most attention?")

attention_cols = st.columns(3)
for i, (_, row) in enumerate(command_df.head(3).iterrows()):
    with attention_cols[i]:
        st.markdown('<div class="city-card">', unsafe_allow_html=True)

        image_path = find_city_image(row["city"])
        if image_path is not None:
            st.image(str(image_path), use_container_width=True)

        st.markdown(f'<div class="city-title">#{int(row["rank"])} {row["city"]}</div>', unsafe_allow_html=True)
        badge(row["readiness_status"], readiness_to_color(row["readiness_status"]))
        st.markdown("")
        st.markdown(f"**Next 24h:** {row['next_24h_level']} ({row['next_24h_score']:.1f})")
        st.markdown(f"**Next 7d peak:** {row['next_7d_peak_level']} ({row['next_7d_peak_score']:.1f})")
        st.markdown(f"**Peak date:** {pd.to_datetime(row['next_7d_peak_date']).strftime('%d.%m.%Y.')}")
        st.markdown(f"**High-risk days:** {int(row['high_risk_days'])}")
        st.markdown(f"**Peak apparent temp:** {row['peak_apparent_temp_max']:.1f} °C")
        st.markdown(f"**Peak ML label:** {row['peak_ml_label']} ({row['peak_ml_confidence']:.2f})")

        if st.button(f"Open Action Center — {row['city']}", key=f"open_action_{row['city']}"):
            st.session_state.selected_city = row["city"]
            st.switch_page("pages/5_Action_Center.py")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Ranking gradova")

rank_fig = px.bar(
    command_df,
    x="city",
    y="priority_score",
    color="readiness_status",
    color_discrete_map={
        "Monitoring": "#2E8B57",
        "Prepared": "#E6A700",
        "Elevated Readiness": "#E67E22",
        "Critical Preparedness": "#C0392B",
    },
    text=command_df["next_7d_peak_level"],
    hover_data={
        "next_24h_level": True,
        "next_24h_score": True,
        "next_72h_peak_level": True,
        "next_72h_peak_score": True,
        "next_7d_peak_level": True,
        "next_7d_peak_score": True,
        "high_risk_days": True,
        "priority_score": True,
        "city": False,
        "readiness_status": False,
    },
    title="City attention ranking",
)
rank_fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis_title="Grad",
    yaxis_title="Priority score",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
rank_fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
st.plotly_chart(rank_fig, use_container_width=True)

st.markdown("## Next 24h vs Next 7d peak")

compare_plot_df = command_df.copy()
compare_plot_df["next_24h_score"] = compare_plot_df["next_24h_score"].round(1)
compare_plot_df["next_7d_peak_score"] = compare_plot_df["next_7d_peak_score"].round(1)

long_df = compare_plot_df.melt(
    id_vars=["city", "readiness_status"],
    value_vars=["next_24h_score", "next_7d_peak_score"],
    var_name="metric",
    value_name="score",
)

long_df["metric"] = long_df["metric"].replace(
    {
        "next_24h_score": "Next 24h score",
        "next_7d_peak_score": "Next 7d peak score",
    }
)

compare_fig = px.bar(
    long_df,
    x="city",
    y="score",
    color="metric",
    barmode="group",
    title="Immediate vs short-term heat risk by city",
)
compare_fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis_title="Grad",
    yaxis_title="Score",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
compare_fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
st.plotly_chart(compare_fig, use_container_width=True)

st.markdown("## Cross-city operational table")

table_df = command_df[
    [
        "rank",
        "city",
        "readiness_status",
        "next_24h_level",
        "next_24h_score",
        "next_72h_peak_level",
        "next_72h_peak_score",
        "next_7d_peak_level",
        "next_7d_peak_score",
        "next_7d_peak_date",
        "high_risk_days",
        "peak_apparent_temp_max",
        "peak_ml_label",
        "peak_ml_confidence",
        "priority_score",
    ]
].copy()

table_df["next_7d_peak_date"] = pd.to_datetime(table_df["next_7d_peak_date"]).dt.strftime("%d.%m.%Y.")
table_df["peak_ml_confidence"] = table_df["peak_ml_confidence"].round(2)

st.dataframe(table_df, use_container_width=True, hide_index=True)

st.markdown("## Attention buckets")

bucket_cols = st.columns(4)

bucket_mapping = {
    "Critical Preparedness": command_df[command_df["readiness_status"] == "Critical Preparedness"]["city"].tolist(),
    "Elevated Readiness": command_df[command_df["readiness_status"] == "Elevated Readiness"]["city"].tolist(),
    "Prepared": command_df[command_df["readiness_status"] == "Prepared"]["city"].tolist(),
    "Monitoring": command_df[command_df["readiness_status"] == "Monitoring"]["city"].tolist(),
}

for idx, (bucket_name, cities_in_bucket) in enumerate(bucket_mapping.items()):
    with bucket_cols[idx]:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{bucket_name}</div>', unsafe_allow_html=True)
        if cities_in_bucket:
            for city in cities_in_bucket:
                st.markdown(f"- {city}")
        else:
            st.markdown("—")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## What to do next")

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Immediate command actions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - fokusirati se prvo na grad #1 u rankingu  
        - otvoriti njegov Action Center  
        - provjeriti next 24h i next 72h peak  
        - aktivirati preporuke za grad, službe i turizam  
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Command dashboard value</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Ovaj modul HeatSafe HR pretvara iz single-city alata u **multi-city smart-city platformu**.
        Time projekt dobiva jači operativni, strateški i natjecateljski karakter.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)