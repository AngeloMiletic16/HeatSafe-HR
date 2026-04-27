from __future__ import annotations

import json
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

MODELS_DIR = PROJECT_ROOT / "data" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

METRICS_V1_PATH = MODELS_DIR / "model_metrics.json"
METRICS_V2_PATH = MODELS_DIR / "model_metrics_strict.json"

FEATURES_V1_PATH = OUTPUTS_DIR / "model_analysis" / "feature_importance.csv"
FEATURES_V2_PATH = OUTPUTS_DIR / "model_analysis_strict" / "feature_importance_strict.csv"

ANALYSIS_V1_PATH = OUTPUTS_DIR / "model_analysis" / "analysis_summary.json"
ANALYSIS_V2_PATH = OUTPUTS_DIR / "model_analysis_strict" / "analysis_summary_strict.json"

st.set_page_config(
    page_title="Methodology & Research",
    page_icon="🧪",
    layout="wide",
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
        padding: 1.45rem 1.6rem 1.25rem 1.6rem;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.16);
    }

    .page-hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .page-hero-subtitle {
        font-size: 1rem;
        line-height: 1.62;
        opacity: 0.95;
        max-width: 1100px;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }

    .chip {
        display: inline-block;
        padding: 0.36rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: white;
        font-size: 0.86rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        min-height: 112px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 0.15rem;
        word-break: break-word;
    }

    .metric-sub {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.55;
    }

    .soft-panel {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
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
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .section-title {
        font-size: 1.36rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.3rem 0 0.85rem 0;
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

    .summary-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        color: #334155;
        line-height: 1.75;
    }

    .summary-card p {
        margin: 0 0 0.9rem 0;
    }

    .summary-card p:last-child {
        margin-bottom: 0;
    }

    .mini-grid-card {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        height: 100%;
    }

    .mini-grid-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.45rem;
    }

    .mini-grid-text {
        color: #475569;
        font-size: 0.93rem;
        line-height: 1.65;
    }

    .stDataFrame, .stPlotlyChart {
        border-radius: 14px;
    }

    div[data-baseweb="tab-list"] {
        gap: 1.15rem;
        margin-top: 0.95rem;
        margin-bottom: 0.85rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
        margin-right: 0.15rem;
    }

    div[data-baseweb="tab-panel"] {
        padding-top: 1rem;
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


@st.cache_data
def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_metric(metrics_json: dict, best_model_name: str, metric_name: str):
    if not isinstance(metrics_json, dict):
        return None
    best_metrics = metrics_json.get(best_model_name, {})
    if not isinstance(best_metrics, dict):
        return None
    return best_metrics.get(metric_name)


def extract_best_metrics(metrics_json: dict) -> dict:
    if not metrics_json:
        return {}

    best_model_name = metrics_json.get("best_model")
    if not best_model_name:
        return {}

    return {
        "best_model": best_model_name,
        "accuracy": safe_metric(metrics_json, best_model_name, "accuracy"),
        "macro_f1": safe_metric(metrics_json, best_model_name, "macro_f1"),
        "weighted_f1": safe_metric(metrics_json, best_model_name, "weighted_f1"),
    }


def build_model_comparison_df(v1: dict, v2: dict) -> pd.DataFrame:
    rows = []

    if v1:
        rows.append(
            {
                "Version": "Production model (v1)",
                "Best model": v1.get("best_model"),
                "Accuracy": v1.get("accuracy"),
                "Macro F1": v1.get("macro_f1"),
                "Weighted F1": v1.get("weighted_f1"),
            }
        )

    if v2:
        rows.append(
            {
                "Version": "Strict model (v2)",
                "Best model": v2.get("best_model"),
                "Accuracy": v2.get("accuracy"),
                "Macro F1": v2.get("macro_f1"),
                "Weighted F1": v2.get("weighted_f1"),
            }
        )

    df = pd.DataFrame(rows)
    for col in ["Accuracy", "Macro F1", "Weighted F1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(float(x), 3) if pd.notna(x) else None)
    return df


def build_feature_chart(df: pd.DataFrame, title: str, top_n: int = 12):
    if df.empty or "feature" not in df.columns or "importance" not in df.columns:
        return None

    top_df = df.head(top_n).copy().sort_values("importance", ascending=True)

    fig = px.bar(
        top_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Importance",
        yaxis_title="Feature",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)")
    fig.update_yaxes(showgrid=False)
    return fig


def fmt_metric(value) -> str:
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


selected_city = st.session_state.get("selected_city", DEFAULT_CITY)
st.session_state.selected_city = selected_city

render_app_sidebar(selected_city=selected_city)

metrics_v1_raw = load_json_if_exists(METRICS_V1_PATH)
metrics_v2_raw = load_json_if_exists(METRICS_V2_PATH)

metrics_v1 = extract_best_metrics(metrics_v1_raw)
metrics_v2 = extract_best_metrics(metrics_v2_raw)

analysis_v1 = load_json_if_exists(ANALYSIS_V1_PATH)
analysis_v2 = load_json_if_exists(ANALYSIS_V2_PATH)

feat_v1 = load_csv_if_exists(FEATURES_V1_PATH)
feat_v2 = load_csv_if_exists(FEATURES_V2_PATH)

comparison_df = build_model_comparison_df(metrics_v1, metrics_v2)

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🧪 Methodology / Research</div>
        <div class="page-hero-subtitle">
            Ova stranica prikazuje metodološku osnovu sustava HeatSafe HR: kako su podaci pripremljeni,
            kako su modeli validirani, zašto postoje production i strict varijanta te kako projekt
            balansira operativnu korisnost, interpretabilnost i istraživačku ozbiljnost.
        </div>
        <div class="chip-row">
            <span class="chip">Applied AI / ML</span>
            <span class="chip">Climate Risk</span>
            <span class="chip">Smart City Research</span>
            <span class="chip">Validation</span>
            <span class="chip">Decision Support</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card(
        "Production model",
        str(metrics_v1.get("best_model", "N/A")),
        f"Macro F1: {fmt_metric(metrics_v1.get('macro_f1'))}",
    )
with k2:
    metric_card(
        "Strict model",
        str(metrics_v2.get("best_model", "N/A")),
        f"Macro F1: {fmt_metric(metrics_v2.get('macro_f1'))}",
    )
with k3:
    metric_card(
        "Test rows (strict)",
        str(analysis_v2.get("test_rows", "N/A")),
        "Time-based evaluation",
    )
with k4:
    metric_card(
        "Research framing",
        "AI + Climate",
        "Smart-city decision support",
    )

st.markdown(
    """
    <div class="note-box">
        <b>Why this page matters:</b> HeatSafe HR nije zamišljen kao vizualni demo bez podloge,
        nego kao ozbiljan AI/ML projekt s jasnim pipelineom, feature engineering logikom,
        validacijom, interpretacijom i dokumentiranim ograničenjima. To je važno i za portfolio
        i za akademski / natjecateljski kredibilitet projekta.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Research framing</div>', unsafe_allow_html=True)

rf1, rf2, rf3 = st.columns(3)
with rf1:
    st.markdown(
        """
        <div class="mini-grid-card">
            <div class="mini-grid-title">Applied problem</div>
            <div class="mini-grid-text">
                Projekt cilja stvarni javni problem: rano prepoznavanje toplinskog rizika u urbanim sredinama
                i prevođenje meteoroloških signala u operativno korisne odluke.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rf2:
    st.markdown(
        """
        <div class="mini-grid-card">
            <div class="mini-grid-title">Methodological value</div>
            <div class="mini-grid-text">
                Sustav ne staje na jednom modelu, nego uspoređuje production i strict pristup
                kako bi se bolje razumjelo koliko performanse dolaze iz pravih obrazaca, a koliko iz shortcut signala.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with rf3:
    st.markdown(
        """
        <div class="mini-grid-card">
            <div class="mini-grid-title">Product relevance</div>
            <div class="mini-grid-text">
                Istraživački sloj nije odvojen od proizvoda: isti modeli hrane forecast, alerting,
                readiness i druge operativne module unutar HeatSafe HR platforme.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(
    [
        "Research Goal",
        "Data & Pipeline",
        "Models & Validation",
        "Heuristic vs ML",
        "Limitations & Future Work",
    ]
)

with tabs[0]:
    st.markdown('<div class="section-title">Research goal</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        panel(
            "Problem statement",
            """
            Cilj projekta HeatSafe HR je rano prepoznavanje toplinskog rizika u hrvatskim gradovima
            pomoću meteoroloških vremenskih nizova, heurističkog risk enginea i modela strojnog učenja.
            Projekt je zamišljen kao decision-support sustav za gradove, javne službe i turistički sektor.
            """,
        )
    with c2:
        panel(
            "Research question",
            """
            Može li se iz povijesnih i forecast meteoroloških signala izgraditi sustav koji:
            <ul>
                <li>detektira i rangira toplinski rizik,</li>
                <li>predviđa buduću razinu rizika,</li>
                <li>podržava operativno odlučivanje kroz readiness i preporuke?</li>
            </ul>
            """,
        )

    st.markdown('<div class="section-title">Scientific contribution</div>', unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        panel(
            "Applied ML contribution",
            """
            Projekt kombinira feature engineering nad vremenskim nizovima, vremenski korektnu validaciju
            i usporedbu više modela za klasifikaciju buduće razine toplinskog rizika.
            """,
        )
    with c4:
        panel(
            "Operational contribution",
            """
            Rezultat nije samo model, nego cijeli sustav: forecast, scenario simulation,
            alerting, command view i operativna interpretacija za krajnjeg korisnika.
            """,
        )
    with c5:
        panel(
            "Societal relevance",
            """
            Tema je relevantna za zdravlje i kvalitetu života, klimatsku otpornost,
            sigurnost, pametne gradove i turizam.
            """,
        )

with tabs[1]:
    st.markdown('<div class="section-title">Data & pipeline</div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        panel(
            "1. Data ingestion",
            """
            Satni meteorološki podaci dohvaćaju se za više hrvatskih gradova.
            Nakon toga se spremaju i organiziraju u strukturirani raw layer.
            """,
        )
    with p2:
        panel(
            "2. Preprocessing",
            """
            Hourly podaci agregiraju se na dnevnu razinu.
            Računaju se osnovne dnevne metrike: temp max/min/mean, humidity, wind i precipitation.
            """,
        )
    with p3:
        panel(
            "3. Risk engine",
            """
            Dnevni meteorološki signali prevode se u heuristički Heat Risk Score i pripadnu razinu rizika.
            Time se dobiva operativni signal razumljiv krajnjim korisnicima.
            """,
        )
    with p4:
        panel(
            "4. Feature engineering",
            """
            Dodaju se lag featurei, rolling prosjeci, persistence signali, kalendarske varijable
            i indikatori toplih/noćnih/vrlo toplih dana za modeliranje sljedećeg dana.
            """,
        )

    st.markdown('<div class="section-title">Data design</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        panel(
            "Input signals",
            """
            <ul>
                <li>temperature (max, min, mean)</li>
                <li>apparent temperature</li>
                <li>humidity</li>
                <li>wind speed</li>
                <li>precipitation</li>
                <li>pressure</li>
                <li>calendar / time features</li>
            </ul>
            """,
        )
    with d2:
        panel(
            "Engineered features",
            """
            <ul>
                <li>lag 1 / lag 2 / lag 3</li>
                <li>rolling 3-day i 7-day mean/max</li>
                <li>hot day i very hot day indikatori</li>
                <li>persistence signali (hot_days_last_3)</li>
                <li>sin/cos sezonske transformacije</li>
            </ul>
            """,
        )

    st.markdown(
        """
        <div class="note-box">
            <b>Bitna metodološka odluka:</b> projekt koristi vremenski smislen pristup.
            Modeli se ne evaluiraju na potpuno nasumičnom splitu, nego na vremenski odvojenom test periodu,
            što bolje simulira stvarnu forecast upotrebu.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[2]:
    st.markdown('<div class="section-title">Models & validation</div>', unsafe_allow_html=True)

    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Model comparison podaci nisu dostupni.")

    st.markdown('<div class="section-title">Zašto 2 modela?</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        panel(
            "Production model (v1)",
            """
            Production model služi kao operativni model sustava.
            Uključuje širi skup signala i koristi se za praktični rad platforme.
            """,
        )
    with c2:
        panel(
            "Strict model (v2)",
            """
            Strict model je metodološki stroža varijanta koja ne koristi
            <code>heat_risk_score*</code> featuree.
            Time se provjerava ima li sustav stvarnu prediktivnu vrijednost
            i bez neposrednog oslanjanja na interno izvedene risk-score varijable.
            """,
        )

    st.markdown('<div class="section-title">Validation logic</div>', unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        panel(
            "Train/test split",
            """
            Korišten je vremenski split: raniji podaci za treniranje,
            noviji period za testiranje. Time se bolje simulira stvarna upotreba modela.
            """,
        )
    with v2:
        panel(
            "Metrics",
            """
            Evaluacija uključuje Accuracy, Macro F1 i Weighted F1.
            Macro F1 je posebno važan jer penalizira lošiju izvedbu na manjim klasama.
            """,
        )
    with v3:
        panel(
            "Interpretability",
            """
            Analizirani su confusion matrix i feature importance.
            Time se dobiva bolji uvid u ponašanje modela i izvore pogrešaka.
            """,
        )

    st.markdown('<div class="section-title">Top featurei</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        fig_v1 = build_feature_chart(feat_v1, "Top featurei — production model", top_n=12)
        if fig_v1 is not None:
            st.plotly_chart(fig_v1, use_container_width=True)
        else:
            st.info("Feature importance za production model nije dostupan.")
    with f2:
        fig_v2 = build_feature_chart(feat_v2, "Top featurei — strict model", top_n=12)
        if fig_v2 is not None:
            st.plotly_chart(fig_v2, use_container_width=True)
        else:
            st.info("Feature importance za strict model nije dostupan.")

    st.markdown(
        """
        <div class="note-box">
            <b>Metodološki zaključak:</b> mali ili umjeren pad performansi između production i strict modela
            sugerira da sustav ne ovisi samo o internom risk scoreu, nego stvarno uči iz meteoroloških obrazaca.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.markdown('<div class="section-title">Heuristic vs ML</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        panel(
            "Heuristic layer",
            """
            Heuristički dio sustava prevodi sirove vremenske signale u razumljiv operativni score i razinu rizika.
            Njegova glavna uloga je:
            <ul>
                <li>transparentnost i interpretabilnost</li>
                <li>scenario simulation</li>
                <li>operativna komunikacija prema krajnjim korisnicima</li>
            </ul>
            """,
        )
    with c2:
        panel(
            "Machine learning layer",
            """
            ML sloj koristi engineered featuree i predviđa buduću klasu rizika.
            Njegova glavna uloga je:
            <ul>
                <li>učenje složenijih obrazaca iz podataka</li>
                <li>predikcija sljedećeg dana</li>
                <li>dodatni signal uz heuristički forecast</li>
            </ul>
            """,
        )

    st.markdown(
        """
        <div class="note-box">
            <b>Zašto kombinacija?</b> Heuristic layer daje interpretabilnost i stabilan operativni signal,
            dok ML layer daje dodatnu prediktivnu inteligenciju. Zajedno čine sustav koji je istodobno
            koristan, objašnjiv i metodološki uvjerljiv.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c3, c4 = st.columns(2)
    with c3:
        panel(
            "Operational advantage",
            """
            U upravljačkom kontekstu korisnicima je važno da vide readiness status, risk level,
            preporuke i scenarije. Zato heuristic sloj ostaje ključan za krajnju upotrebu.
            """,
        )
    with c4:
        panel(
            "Research advantage",
            """
            Za akademsku i tehničku ozbiljnost važno je pokazati da model daje dodatnu vrijednost.
            Zato su uključeni strict model, vremenska validacija i analiza feature importance.
            """,
        )

with tabs[4]:
    st.markdown('<div class="section-title">Limitations & future work</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        panel(
            "Current limitations",
            """
            <ul>
                <li>ograničen broj gradova i geografska pokrivenost</li>
                <li>oslanjanje na dostupne meteorološke podatke i proxy signale</li>
                <li>trenutno nema direktne integracije sa zdravstvenim ili intervencijskim podacima</li>
                <li>risk engine je heuristički konstruiran i može se dodatno kalibrirati</li>
            </ul>
            """,
        )
    with c2:
        panel(
            "Next improvements",
            """
            <ul>
                <li>širenje na više gradova i regija</li>
                <li>bolji forecast modeli i dulji horizont predikcije</li>
                <li>kalibracija risk score sustava s dodatnim ekspertizama</li>
                <li>integracija mobilnih notifikacija i alertinga</li>
                <li>povezivanje s realnim urbanim i zdravstvenim indikatorima</li>
            </ul>
            """,
        )

    st.markdown(
        """
        <div class="warning-box">
            <b>Važna napomena:</b> HeatSafe HR je decision-support sustav, a ne zamjena za službene meteorološke,
            zdravstvene ili krizne protokole. Njegova uloga je rano upozorenje, procjena i podrška odlučivanju.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Research positioning</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        panel(
            "Mentor / faculty view",
            """
            Projekt pokazuje data engineering, ML modeliranje, validaciju,
            interpretaciju i product thinking unutar jedne povezane platforme.
            """,
        )
    with r2:
        panel(
            "Award view",
            """
            Projekt kombinira AI/ML, smart city, climate resilience,
            sigurnost i kvalitetu života u vrlo jasnom društveno korisnom okviru.
            """,
        )
    with r3:
        panel(
            "Portfolio view",
            """
            Projekt pokazuje da kandidat može izgraditi cijeli sustav:
            data pipeline + model + evaluaciju + aplikaciju + decision layer.
            """,
        )

st.divider()

st.markdown('<div class="section-title">Methodology summary</div>', unsafe_allow_html=True)

strict_test_rows = analysis_v2.get("test_rows", "N/A")
strict_train_rows = analysis_v2.get("train_rows", "N/A")

summary_html = (
    f'<div class="summary-card">'
    f'<p>HeatSafe HR metodološki je postavljen kao <b>applied AI/ML decision-support projekt</b>, '
    f'a ne samo kao vizualni dashboard. U središtu sustava su vremenski podaci, feature engineering nad vremenskim nizovima, '
    f'heuristički risk engine i usporedba production i strict modelne varijante.</p>'
    f'<p>Production model služi operativnoj upotrebi platforme, dok strict model daje dodatni research sloj '
    f'jer provjerava koliko sustav zadržava prediktivnu vrijednost bez oslanjanja na shortcut featuree. '
    f'U trenutno dokumentiranoj strict evaluaciji sustav raspolaže s <b>{strict_train_rows}</b> train redaka i '
    f'<b>{strict_test_rows}</b> test redaka.</p>'
    f'<p>Takva arhitektura daje projektu jaču akademsku težinu, veću transparentnost i bolju podlogu '
    f'za natjecateljski, portfolio i budući istraživački razvoj.</p>'
    f'</div>'
)

st.markdown(summary_html, unsafe_allow_html=True)