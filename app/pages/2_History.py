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
from src.sidebar import render_app_sidebar

RISK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_cities_daily_with_risk.csv"
CITY_IMAGE_DIR = PROJECT_ROOT / "app" / "assets" / "cities"

RISK_ORDER = ["Nizak", "Umjeren", "Visok", "Vrlo visok"]
RISK_COLOR_MAP = {
    "Nizak": "#2E8B57",
    "Umjeren": "#E6A700",
    "Visok": "#E67E22",
    "Vrlo visok": "#C0392B",
}
READINESS_MAP = {
    "Nizak": "Monitoring",
    "Umjeren": "Prepared",
    "Visok": "Elevated Readiness",
    "Vrlo visok": "Critical Preparedness",
}

st.set_page_config(
    page_title="History | HeatSafe HR",
    page_icon="🕘",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


@st.cache_data
def load_risk_data() -> pd.DataFrame:
    if not RISK_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RISK_DATA_PATH}. Run risk engine first.")

    df = pd.read_csv(RISK_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    return df


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0b3b2e 100%);
            border-radius: 22px;
            padding: 1.55rem 1.7rem 1.35rem 1.7rem;
            color: white;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1rem;
            line-height: 1.65;
            opacity: 0.95;
            margin-bottom: 0.95rem;
            max-width: 1050px;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.35rem;
        }

        .chip {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: white;
            font-size: 0.88rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.12);
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            min-height: 118px;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #475569;
            font-weight: 600;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }

        .metric-sub {
            font-size: 0.88rem;
            color: #64748b;
            line-height: 1.5;
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0.35rem 0 0.85rem 0;
            color: #0f172a;
        }

        .card {
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
            height: 100%;
        }

        .soft-note {
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 0.95rem 1rem;
            border-radius: 12px;
            color: #0f172a;
            margin-top: 0.6rem;
            margin-bottom: 0.7rem;
            line-height: 1.65;
        }

        .warning-note {
            background: #fff7ed;
            border-left: 5px solid #f97316;
            padding: 0.95rem 1rem;
            border-radius: 12px;
            color: #7c2d12;
            margin-top: 0.6rem;
            margin-bottom: 0.7rem;
            line-height: 1.65;
        }

        .status-pill {
            display: inline-block;
            padding: 0.42rem 0.82rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.94rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .mini-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .big-text {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }

        .small-muted {
            font-size: 0.9rem;
            color: #64748b;
            line-height: 1.6;
        }

        .closing-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        }

        .stDataFrame, .stPlotlyChart {
            border-radius: 14px;
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

    exact_candidates = [
        CITY_IMAGE_DIR / f"{city_name}.jpg",
        CITY_IMAGE_DIR / f"{city_name}.jpeg",
        CITY_IMAGE_DIR / f"{city_name}.png",
        CITY_IMAGE_DIR / f"{city_name}.webp",
    ]
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = CITY_IMAGE_DIR / f"{slug}{ext}"
        if candidate.exists():
            return candidate

    return None


def render_metric_card(label: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{label}</span>',
        unsafe_allow_html=True,
    )


def risk_to_readiness(risk_level: str) -> str:
    return READINESS_MAP.get(risk_level, "Monitoring")


def build_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            avg_heat_risk_score=("heat_risk_score", "mean"),
            max_heat_risk_score=("heat_risk_score", "max"),
            high_risk_days=("risk_level", lambda s: ((s == "Visok") | (s == "Vrlo visok")).sum()),
            very_high_days=("risk_level", lambda s: (s == "Vrlo visok").sum()),
            max_apparent_temp=("apparent_temp_max", "max"),
        )
    )
    return yearly.sort_values("year")


def build_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["month", "month_name"], as_index=False)
        .agg(
            avg_heat_risk_score=("heat_risk_score", "mean"),
            avg_temp_max=("temp_max", "mean"),
            avg_apparent_temp_max=("apparent_temp_max", "mean"),
            high_risk_days=("risk_level", lambda s: ((s == "Visok") | (s == "Vrlo visok")).sum()),
        )
        .sort_values("month")
    )
    return monthly


def build_heatwave_candidates(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["high_risk_flag"] = (out["heat_risk_score"] >= threshold).astype(int)

    groups = []
    current_group = []

    for _, row in out.iterrows():
        if row["high_risk_flag"] == 1:
            current_group.append(row)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    if current_group:
        groups.append(current_group)

    records = []
    for group in groups:
        if len(group) >= 2:
            gdf = pd.DataFrame(group)
            records.append(
                {
                    "start_date": gdf["date"].min(),
                    "end_date": gdf["date"].max(),
                    "duration_days": len(gdf),
                    "max_heat_risk_score": gdf["heat_risk_score"].max(),
                    "max_apparent_temp": gdf["apparent_temp_max"].max(),
                    "avg_temp_max": gdf["temp_max"].mean(),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "duration_days",
                "max_heat_risk_score",
                "max_apparent_temp",
                "avg_temp_max",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["duration_days", "max_heat_risk_score"],
        ascending=[False, False],
    )


inject_custom_css()

df = load_risk_data()
cities = sorted(df["city"].unique().tolist())
default_index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0

if "selected_city" not in st.session_state:
    st.session_state.selected_city = DEFAULT_CITY if DEFAULT_CITY in cities else cities[0]

control_left, control_right = st.columns([1, 1])
with control_left:
    selected_city = st.selectbox(
        "Odaberi grad",
        cities,
        index=cities.index(st.session_state.selected_city)
        if st.session_state.selected_city in cities
        else default_index,
    )
with control_right:
    selected_year_mode = st.selectbox(
        "Povijesni raspon",
        ["Sve godine", "Zadnje 3 godine", "Zadnja 1 godina"],
        index=0,
    )

st.session_state.selected_city = selected_city

city_df_full = df[df["city"] == selected_city].sort_values("date").copy()
latest_row = city_df_full.sort_values("date").iloc[-1]

render_app_sidebar(
    selected_city=selected_city,
    risk_level=latest_row["risk_level"],
    readiness_status=risk_to_readiness(str(latest_row["risk_level"])),
)

city_df = city_df_full.copy()
if selected_year_mode == "Zadnje 3 godine":
    cutoff = city_df["date"].max() - pd.Timedelta(days=365 * 3)
    city_df = city_df[city_df["date"] >= cutoff].copy()
elif selected_year_mode == "Zadnja 1 godina":
    cutoff = city_df["date"].max() - pd.Timedelta(days=365)
    city_df = city_df[city_df["date"] >= cutoff].copy()

if city_df.empty:
    st.warning("Nema podataka za odabrani period.")
    st.stop()

yearly_df = build_yearly_summary(city_df)
monthly_df = build_monthly_climatology(city_df)
episodes_df = build_heatwave_candidates(city_df, threshold=50.0)

latest_date = pd.to_datetime(city_df["date"]).max().strftime("%d.%m.%Y.")
period_start = pd.to_datetime(city_df["date"]).min().strftime("%d.%m.%Y.")
period_end = pd.to_datetime(city_df["date"]).max().strftime("%d.%m.%Y.")
city_image_path = find_city_image(selected_city)

st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">🕘 Historical Risk Intelligence</div>
        <div class="hero-subtitle">
            Povijesni modul za grad <b>{selected_city}</b> prikazuje kako su se toplinski rizici
            razvijali kroz godine, mjesece i najizraženije toplinske epizode. Ovaj pogled pomaže
            razumjeti sezonske obrasce, prepoznati kritične periode i dati gradu bolju osnovu za
            ljetnu pripremu, readiness planiranje i dugoročnije climate-resilience odluke.
        </div>
        <div class="chip-row">
            <span class="chip">Historical Intelligence</span>
            <span class="chip">Seasonality</span>
            <span class="chip">Heat Episodes</span>
            <span class="chip">City Context</span>
            <span class="chip">Risk Memory</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.35, 0.9])
with hero_left:
    st.markdown(
        f"""
        <div class="soft-note">
            <b>Historical framing:</b> ova stranica nije samo pregled prošlih temperatura, nego
            city-level povijesni risk memory sloj. Koristi se za razumijevanje kada grad tipično
            ulazi u opasniji toplinski režim, koliko dugo traju kritične epizode i kako se
            intenzitet rizika mijenja kroz sezonu.<br><br>
            <b>Selected period:</b> {period_start} — {period_end} &nbsp; | &nbsp;
            <b>Latest available day:</b> {latest_date}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_year_mode != "Sve godine":
        st.markdown(
            f"""
            <div class="warning-note">
                <b>Filtered view active:</b> koristiš način prikaza <b>{selected_year_mode}</b>.
                Ovaj pogled je dobar za recentnu operativnu usporedbu, ali za puni klimatski kontekst
                preporučuje se i pregled svih dostupnih godina.
            </div>
            """,
            unsafe_allow_html=True,
        )

with hero_right:
    if city_image_path is not None:
        st.image(str(city_image_path), use_container_width=True)
    else:
        st.markdown(
            f"""
            <div class="card">
                <div class="mini-title">City context</div>
                <div class="big-text">{selected_city}</div>
                <div class="small-muted">
                    Povijesni profil toplinskog rizika prikazuje kako se grad ponaša kroz sezonu,
                    koliko ima visokorizičnih dana i kakve su mu višednevne toplinske epizode.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric_card(
        "Days in dataset",
        f"{len(city_df):,}",
        "Historical records in selected view",
    )
with metric_cols[1]:
    render_metric_card(
        "Avg Heat Risk Score",
        f"{city_df['heat_risk_score'].mean():.1f}",
        "Mean daily heat-risk intensity",
    )
with metric_cols[2]:
    render_metric_card(
        "Max apparent temp",
        f"{city_df['apparent_temp_max'].max():.1f} °C",
        "Highest observed apparent temperature",
    )
with metric_cols[3]:
    render_metric_card(
        "High-risk days",
        str(int(((city_df["risk_level"] == "Visok") | (city_df["risk_level"] == "Vrlo visok")).sum())),
        "High and very high risk days",
    )

st.markdown('<div class="section-title">Historical command summary</div>', unsafe_allow_html=True)

summary_left, summary_right = st.columns([1.05, 1])

highest_year = int(yearly_df.sort_values("max_apparent_temp", ascending=False).iloc[0]["year"])
highest_year_score = int(yearly_df.sort_values("avg_heat_risk_score", ascending=False).iloc[0]["year"])
peak_month_row = monthly_df.sort_values("avg_heat_risk_score", ascending=False).iloc[0]
episodes_count = len(episodes_df)

with summary_left:
    st.markdown(
        f"""
        <div class="card">
            <div class="mini-title">Historical interpretation</div>
            <div class="small-muted">
                Za grad <b>{selected_city}</b> najizraženiji toplinski signal u odabranom povijesnom
                prikazu povezan je s godinom <b>{highest_year_score}</b>, dok je najviša apparent
                temperatura zabilježena u godini <b>{highest_year}</b>. Sezonski vrhunac prosječnog
                Heat Risk Score trenutačno pada u mjesec <b>{peak_month_row['month_name']}</b>, što
                ovaj grad čini posebno važnim za raniju ljetnu pripremu i monitoring u tom dijelu godine.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_right:
    st.markdown(
        f"""
        <div class="card">
            <div class="mini-title">What history tells operators</div>
            <div class="small-muted">
                Povijesni sloj pomaže razlikovati kratkotrajne vruće dane od stvarnih višednevnih
                toplinskih epizoda. U ovom prikazu sustav je identificirao <b>{episodes_count}</b>
                kandidata za višednevne toplinske epizode s izraženijim score signalom, što je korisno
                za usporedbu s aktualnim forecastom, seasonal readiness planiranjem i gradskim
                preventivnim mjerama.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown('<div class="section-title">Yearly risk profile</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    fig_yearly = px.bar(
        yearly_df,
        x="year",
        y="avg_heat_risk_score",
        title=f"Godišnji prosječni Heat Risk Score — {selected_city}",
        text_auto=".1f",
    )
    fig_yearly.update_layout(
        xaxis_title="Godina",
        yaxis_title="Prosječni score",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_yearly.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_yearly, use_container_width=True)

with right:
    fig_high_days = px.bar(
        yearly_df,
        x="year",
        y="high_risk_days",
        title=f"Visoki i vrlo visoki rizik po godini — {selected_city}",
        text_auto=True,
    )
    fig_high_days.update_layout(
        xaxis_title="Godina",
        yaxis_title="Broj dana",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_high_days.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_high_days, use_container_width=True)

st.divider()

st.markdown('<div class="section-title">Seasonality and monthly climatology</div>', unsafe_allow_html=True)

month_order = monthly_df["month_name"].tolist()

col_month_1, col_month_2 = st.columns(2)

with col_month_1:
    fig_monthly_risk = px.line(
        monthly_df,
        x="month_name",
        y="avg_heat_risk_score",
        markers=True,
        category_orders={"month_name": month_order},
        title=f"Mjesečni prosjeci Heat Risk Score — {selected_city}",
    )
    fig_monthly_risk.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="Prosječni score",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_monthly_risk.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_monthly_risk, use_container_width=True)

with col_month_2:
    fig_monthly_temp = px.line(
        monthly_df,
        x="month_name",
        y="avg_apparent_temp_max",
        markers=True,
        category_orders={"month_name": month_order},
        title=f"Mjesečni prosjeci apparent temperature — {selected_city}",
    )
    fig_monthly_temp.update_layout(
        xaxis_title="Mjesec",
        yaxis_title="°C",
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_monthly_temp.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
    st.plotly_chart(fig_monthly_temp, use_container_width=True)

st.divider()

st.markdown('<div class="section-title">Daily historical heat-risk timeline</div>', unsafe_allow_html=True)

fig_daily = px.line(
    city_df,
    x="date",
    y="heat_risk_score",
    color="risk_level",
    category_orders={"risk_level": RISK_ORDER},
    color_discrete_map=RISK_COLOR_MAP,
    title=f"Dnevni Heat Risk Score kroz vrijeme — {selected_city}",
)
fig_daily.update_layout(
    xaxis_title="Datum",
    yaxis_title="Heat Risk Score",
    margin=dict(l=20, r=20, t=55, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="Risk level",
)
fig_daily.update_yaxes(gridcolor="rgba(15,23,42,0.08)")
st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

st.markdown('<div class="section-title">Historical episodes and critical days</div>', unsafe_allow_html=True)

col_table_1, col_table_2 = st.columns(2)

with col_table_1:
    st.markdown("### Najizraženije toplinske epizode")
    if episodes_df.empty:
        st.info("Nisu pronađene višednevne epizode s Heat Risk Score ≥ 50.")
    else:
        episodes_show = episodes_df.head(15).copy()
        episodes_show["start_date"] = pd.to_datetime(episodes_show["start_date"]).dt.strftime("%d.%m.%Y.")
        episodes_show["end_date"] = pd.to_datetime(episodes_show["end_date"]).dt.strftime("%d.%m.%Y.")
        episodes_show["avg_temp_max"] = episodes_show["avg_temp_max"].round(1)
        episodes_show["max_heat_risk_score"] = episodes_show["max_heat_risk_score"].round(1)
        episodes_show["max_apparent_temp"] = episodes_show["max_apparent_temp"].round(1)
        st.dataframe(episodes_show, use_container_width=True, hide_index=True)

with col_table_2:
    st.markdown("### Top 15 dana po najvećem riziku")
    top_days = (
        city_df.sort_values(["heat_risk_score", "apparent_temp_max"], ascending=[False, False])[
            ["date", "risk_level", "heat_risk_score", "temp_max", "apparent_temp_max", "humidity_mean"]
        ]
        .head(15)
        .copy()
    )
    top_days["date"] = pd.to_datetime(top_days["date"]).dt.strftime("%d.%m.%Y.")
    top_days["heat_risk_score"] = top_days["heat_risk_score"].round(1)
    top_days["temp_max"] = top_days["temp_max"].round(1)
    top_days["apparent_temp_max"] = top_days["apparent_temp_max"].round(1)
    top_days["humidity_mean"] = top_days["humidity_mean"].round(1)
    st.dataframe(top_days, use_container_width=True, hide_index=True)

st.divider()

st.markdown('<div class="section-title">Historical closing summary</div>', unsafe_allow_html=True)

closing_left, closing_right = st.columns([1.25, 1])

with closing_left:
    st.markdown(
        f"""
        <div class="closing-card">
            <div class="mini-title">Operator takeaway</div>
            <div class="big-text">{selected_city} historical heat profile</div>
            <div class="small-muted">
                Povijesni pregled pokazuje da je za grad <b>{selected_city}</b> najviši apparent-temperature
                ekstrem zabilježen u godini <b>{highest_year}</b>, dok je najveći prosječni Heat Risk Score
                ostvaren u godini <b>{highest_year_score}</b>. Ovakav historical layer daje važan kontekst
                za interpretaciju aktualnih forecast signala, pomaže u seasonal readiness planiranju i
                jača vjerodostojnost platforme kao ozbiljnog decision-support alata, a ne samo dashboarda.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with closing_right:
    st.markdown(
        f"""
        <div class="closing-card">
            <div class="mini-title">Why this page matters</div>
            <div class="small-muted">
                Ova stranica pretvara sirovu povijest vremena u <b>operational memory</b> sustava.
                Time HeatSafe HR može pokazati ne samo što se događa danas, nego i kako se grad ponašao
                u prethodnim sezonama, kada ulazi u kritične obrasce i koje dijelove ljetne sezone treba
                promatrati kao prioritetne za preventivni odgovor.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )