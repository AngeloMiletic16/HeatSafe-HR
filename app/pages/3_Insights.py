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
from src.forecast_engine import make_ml_forecast
from src.xai_engine import explain_escalation_row

st.set_page_config(page_title="Insights", page_icon="🧠", layout="wide")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

MODEL_ANALYSIS_DIR = OUTPUTS_DIR / "model_analysis"
MODEL_ANALYSIS_STRICT_DIR = OUTPUTS_DIR / "model_analysis_strict"
MODEL_ANALYSIS_ESCALATION_DIR = OUTPUTS_DIR / "model_analysis_escalation"

METRICS_V1_PATH = MODELS_DIR / "model_metrics.json"
METRICS_V2_PATH = MODELS_DIR / "model_metrics_strict.json"
METRICS_V3_PATH = MODELS_DIR / "model_metrics_escalation.json"

CONFUSION_V1_PATH = MODEL_ANALYSIS_DIR / "confusion_matrix.csv"
CONFUSION_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "confusion_matrix_strict.csv"
CONFUSION_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "confusion_matrix_escalation_detailed.csv"

FEATURES_V1_PATH = MODEL_ANALYSIS_DIR / "feature_importance.csv"
FEATURES_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "feature_importance_strict.csv"
FEATURES_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "feature_importance_escalation.csv"

REPORT_V1_PATH = MODEL_ANALYSIS_DIR / "classification_report.csv"
REPORT_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "classification_report_strict.csv"
REPORT_V3_JSON_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "classification_report_escalation.json"

PRED_V1_PATH = MODEL_ANALYSIS_DIR / "test_predictions_detailed.csv"
PRED_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "test_predictions_detailed_strict.csv"
PRED_V3_PATH = MODELS_DIR / "test_predictions_escalation.csv"

FALSE_POS_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "false_positives_escalation.csv"
FALSE_NEG_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "false_negatives_escalation.csv"

THRESHOLD_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "threshold_tuning_escalation.csv"
THRESHOLD_SUMMARY_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "threshold_tuning_summary.json"
ANALYSIS_SUMMARY_V3_PATH = MODEL_ANALYSIS_ESCALATION_DIR / "analysis_summary_escalation_detailed.json"

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=index_col)


def extract_best_metrics_multiclass(metrics_json: dict) -> dict:
    if not metrics_json:
        return {}
    best_model_name = metrics_json.get("best_model")
    if not best_model_name:
        return {}
    best_metrics = metrics_json.get(best_model_name, {})
    return {
        "best_model": best_model_name,
        "accuracy": best_metrics.get("accuracy"),
        "macro_f1": best_metrics.get("macro_f1"),
        "weighted_f1": best_metrics.get("weighted_f1"),
    }


def extract_best_metrics_escalation(metrics_json: dict) -> dict:
    if not metrics_json:
        return {}

    best_model_name = metrics_json.get("best_model")
    if not best_model_name:
        return {}

    models_block = metrics_json.get("models", {})
    if not isinstance(models_block, dict):
        return {"best_model": best_model_name}

    best_metrics = models_block.get(best_model_name, {})
    if not isinstance(best_metrics, dict):
        return {"best_model": best_model_name}

    return {
        "best_model": best_model_name,
        "accuracy": best_metrics.get("accuracy"),
        "balanced_accuracy": best_metrics.get("balanced_accuracy"),
        "precision_positive": best_metrics.get("precision_positive"),
        "recall_positive": best_metrics.get("recall_positive"),
        "f1_positive": best_metrics.get("f1_positive"),
        "roc_auc": best_metrics.get("roc_auc"),
    }

def build_model_comparison_df(v1: dict, v2: dict, v3: dict) -> pd.DataFrame:
    rows = []

    if v1:
        rows.append(
            {
                "Version": "Production model (v1)",
                "Task": "Next-day multiclass heat-risk classification",
                "Best model": v1.get("best_model"),
                "Accuracy": v1.get("accuracy"),
                "Macro F1": v1.get("macro_f1"),
                "Weighted F1": v1.get("weighted_f1"),
                "Positive F1": None,
                "ROC AUC": None,
            }
        )

    if v2:
        rows.append(
            {
                "Version": "Strict model (v2)",
                "Task": "Methodologically stricter multiclass validation",
                "Best model": v2.get("best_model"),
                "Accuracy": v2.get("accuracy"),
                "Macro F1": v2.get("macro_f1"),
                "Weighted F1": v2.get("weighted_f1"),
                "Positive F1": None,
                "ROC AUC": None,
            }
        )

    if v3:
        rows.append(
            {
                "Version": "Escalation model (v3)",
                "Task": "72h escalation early-warning classification",
                "Best model": v3.get("best_model"),
                "Accuracy": v3.get("accuracy"),
                "Macro F1": None,
                "Weighted F1": None,
                "Positive F1": v3.get("f1_positive"),
                "ROC AUC": v3.get("roc_auc"),
            }
        )

    return pd.DataFrame(rows)


def build_confusion_heatmap(df: pd.DataFrame, title: str):
    if df.empty:
        return None

    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return fig


def build_top_features_chart(df: pd.DataFrame, title: str, top_n: int = 15):
    if df.empty:
        return None

    importance_col = None
    for col in ["importance", "mean_abs_shap", "value"]:
        if col in df.columns:
            importance_col = col
            break

    feature_col = None
    for col in ["feature", "Feature"]:
        if col in df.columns:
            feature_col = col
            break

    if importance_col is None or feature_col is None:
        return None

    top_df = df[[feature_col, importance_col]].copy().head(top_n)
    top_df = top_df.sort_values(importance_col, ascending=True)

    fig = px.bar(
        top_df,
        x=importance_col,
        y=feature_col,
        orientation="h",
        title=title,
    )
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_error_examples(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "correct_prediction" not in df.columns:
        return pd.DataFrame()

    errors = df[df["correct_prediction"] == 0].copy()
    if errors.empty:
        return errors

    cols = [
        "city",
        "date",
        "true_label",
        "predicted_label",
        "heat_risk_score",
    ]
    keep_cols = [c for c in cols if c in errors.columns]
    errors = errors[keep_cols].copy()

    if "date" in errors.columns:
        errors["date"] = pd.to_datetime(errors["date"]).dt.strftime("%d.%m.%Y.")

    return errors.head(25)


def build_class_report_chart(report_df: pd.DataFrame, title: str):
    if report_df.empty or "Unnamed: 0" not in report_df.columns:
        return None

    class_rows = report_df[
        report_df["Unnamed: 0"].isin(["Nizak", "Umjeren", "Visok", "Vrlo visok"])
    ].copy()

    if class_rows.empty:
        return None

    melted = class_rows.melt(
        id_vars="Unnamed: 0",
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="value",
    )

    fig = px.bar(
        melted,
        x="Unnamed: 0",
        y="value",
        color="metric",
        barmode="group",
        title=title,
    )
    fig.update_layout(
        xaxis_title="Klasa",
        yaxis_title="Vrijednost",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_v3_threshold_chart(df: pd.DataFrame):
    if df.empty:
        return None

    metric_cols = [c for c in ["precision_positive", "recall_positive", "f1_positive", "accuracy"] if c in df.columns]
    if not metric_cols or "threshold" not in df.columns:
        return None

    melted = df.melt(
        id_vars="threshold",
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    fig = px.line(
        melted,
        x="threshold",
        y="value",
        color="metric",
        markers=True,
        title="v3 threshold tuning",
    )
    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Score",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_v3_classification_summary_df(report_json: dict) -> pd.DataFrame:
    if not report_json:
        return pd.DataFrame()

    rows = []
    for label in ["0", "1"]:
        if label in report_json:
            rows.append(
                {
                    "Class": label,
                    "Precision": report_json[label].get("precision"),
                    "Recall": report_json[label].get("recall"),
                    "F1-score": report_json[label].get("f1-score"),
                    "Support": report_json[label].get("support"),
                }
            )

    return pd.DataFrame(rows)


def build_v3_error_examples(df: pd.DataFrame, title_type: str) -> pd.DataFrame:
    if df.empty:
        return df

    preferred_cols = [
        "city",
        "date",
        "actual_escalation_72h",
        "predicted_escalation_72h",
        "predicted_escalation_probability",
    ]
    keep_cols = [c for c in preferred_cols if c in df.columns]
    out = df[keep_cols].copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d.%m.%Y.")

    if "actual_escalation_72h" in out.columns:
        out = out.rename(columns={"actual_escalation_72h": "actual"})
    if "predicted_escalation_72h" in out.columns:
        out = out.rename(columns={"predicted_escalation_72h": "predicted"})
    if "predicted_escalation_probability" in out.columns:
        out = out.rename(columns={"predicted_escalation_probability": "probability"})

    return out.head(25)


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


def render_list_card(title: str, items: list[str], subtitle: str = "") -> None:
    list_html = "".join(f"<li>{item}</li>" for item in items)
    subtitle_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">{title}</div>
            {subtitle_html}
            <ul class="info-list">
                {list_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1400px;
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
        line-height: 1.6;
        opacity: 0.95;
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
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 0.15rem;
    }

    .metric-sub {
        font-size: 0.88rem;
        color: #64748b;
    }

    .info-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        height: 100%;
    }

    .card-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.65rem;
    }

    .card-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.65rem;
        line-height: 1.55;
    }

    .info-list {
        margin: 0;
        padding-left: 1.1rem;
        color: #334155;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    .note-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 0.95rem 1rem;
        border-radius: 12px;
        color: #0f172a;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
        line-height: 1.6;
    }

    div[data-baseweb="tab-list"] {
        gap: 1.2rem;
        margin-top: 0.9rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    button[data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-hero">
        <div class="page-hero-title">🧠 Insights / AI & Research Layer</div>
        <div class="page-hero-subtitle">
            Ova stranica prikazuje AI/ML srce sustava HeatSafe HR: usporedbu modela,
            confusion matrix, feature importance, threshold tuning, tipične greške
            i explainable AI sloj za v3 escalation model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Load data ----
metrics_v1_raw = load_json(METRICS_V1_PATH)
metrics_v2_raw = load_json(METRICS_V2_PATH)
metrics_v3_raw = load_json(METRICS_V3_PATH)


metrics_v1 = extract_best_metrics_multiclass(metrics_v1_raw)
metrics_v2 = extract_best_metrics_multiclass(metrics_v2_raw)
metrics_v3 = extract_best_metrics_escalation(metrics_v3_raw)

comparison_df = build_model_comparison_df(metrics_v1, metrics_v2, metrics_v3)

conf_v1 = load_csv(CONFUSION_V1_PATH, index_col=0)
conf_v2 = load_csv(CONFUSION_V2_PATH, index_col=0)
conf_v3 = load_csv(CONFUSION_V3_PATH, index_col=0)

feat_v1 = load_csv(FEATURES_V1_PATH)
feat_v2 = load_csv(FEATURES_V2_PATH)
feat_v3 = load_csv(FEATURES_V3_PATH)

report_v1 = load_csv(REPORT_V1_PATH)
report_v2 = load_csv(REPORT_V2_PATH)
report_v3_json = load_json(REPORT_V3_JSON_PATH)
report_v3_df = build_v3_classification_summary_df(report_v3_json)

pred_v1 = load_csv(PRED_V1_PATH)
pred_v2 = load_csv(PRED_V2_PATH)
pred_v3 = load_csv(PRED_V3_PATH)

false_pos_v3 = load_csv(FALSE_POS_V3_PATH)
false_neg_v3 = load_csv(FALSE_NEG_V3_PATH)

threshold_v3 = load_csv(THRESHOLD_V3_PATH)
threshold_summary_v3 = load_json(THRESHOLD_SUMMARY_V3_PATH)
analysis_summary_v3 = load_json(ANALYSIS_SUMMARY_V3_PATH)

# ---- Top controls for live XAI ----
default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = CITIES.index(default_city) if default_city in CITIES else 0

top1, top2 = st.columns([1, 1])

with top1:
    selected_city = st.selectbox("Odaberi grad za live XAI demo", CITIES, index=default_index)
    st.session_state.selected_city = selected_city

with top2:
    xai_scenario_enabled = st.toggle("Scenario mode za XAI demo", value=True)

if xai_scenario_enabled:
    s1, s2, s3 = st.columns(3)
    with s1:
        xai_temperature_delta = st.slider("Promjena temperature (°C)", -2, 12, 6, 1, key="xai_temp_delta")
    with s2:
        xai_humidity_delta = st.slider("Promjena vlage (%)", -20, 30, 10, 1, key="xai_humidity_delta")
    with s3:
        xai_wind_delta = st.slider("Promjena vjetra (m/s)", -8, 5, -3, 1, key="xai_wind_delta")
else:
    xai_temperature_delta = 0
    xai_humidity_delta = 0
    xai_wind_delta = 0

try:
    xai_forecast_df = make_ml_forecast(
        selected_city,
        temperature_delta=xai_temperature_delta,
        humidity_delta=xai_humidity_delta,
        wind_delta=xai_wind_delta,
    )
    xai_input_row = xai_forecast_df.sort_values("date").head(1).copy()
    if "city" not in xai_input_row.columns:
        xai_input_row["city"] = selected_city
    xai_summary = explain_escalation_row(xai_input_row)
except Exception as exc:
    xai_forecast_df = pd.DataFrame()
    xai_input_row = pd.DataFrame()
    xai_summary = {
        "method": "error",
        "probability": None,
        "label": None,
        "top_positive_drivers": [],
        "top_protective_drivers": [],
        "explanation_text": f"XAI demo nije dostupan: {exc}",
    }

# ---- Summary cards ----
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card(
        "Production Macro F1",
        f"{metrics_v1.get('macro_f1', 'N/A')}",
        str(metrics_v1.get("best_model", "N/A")),
    )
with k2:
    render_metric_card(
        "Strict Macro F1",
        f"{metrics_v2.get('macro_f1', 'N/A')}",
        str(metrics_v2.get("best_model", "N/A")),
    )
with k3:
    render_metric_card(
        "Escalation F1+",
        f"{metrics_v3.get('f1_positive', 'N/A')}",
        str(metrics_v3.get("best_model", "N/A")),
    )
with k4:
    render_metric_card(
        "Escalation ROC AUC",
        f"{metrics_v3.get('roc_auc', 'N/A')}",
        "v3 early-warning model",
    )

st.markdown(
    """
    <div class="note-box">
        HeatSafe HR sada više nije samo weather dashboard. 
        Sustav kombinira multiclass procjenu rizika (v1, v2), 72h escalation early-warning model (v3),
        threshold tuning, explainable AI i operativne briefove za stvarnu decision-support upotrebu.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Model comparison",
        "Production vs Strict",
        "Escalation model v3",
        "Live XAI demo",
    ]
)

# ---- Tab 1: comparison ----
with tabs[0]:
    st.markdown("## Usporedba modela")

    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Model comparison podaci nisu dostupni.")

    c1, c2 = st.columns(2)
    with c1:
        render_list_card(
            "Što predstavljaju modeli",
            [
                "Production model (v1) služi za operativnu multiclass procjenu toplinskog rizika.",
                "Strict model (v2) daje metodološki strožu validaciju bez oslanjanja na risk-score featuree.",
                "Escalation model (v3) predviđa postoji li vjerojatna eskalacija u sljedeća 72 sata.",
            ],
            "Tri modelna sloja zajedno dižu vjerodostojnost i korisnost sustava.",
        )

    with c2:
        render_list_card(
            "Zašto je ovo važno za projekt",
            [
                "v1 daje praktičnu operativnu vrijednost.",
                "v2 jača research credibility projekta.",
                "v3 dodaje early-warning dimenziju i diže decision-support kvalitetu.",
            ],
            "Ova kombinacija pokazuje da projekt nije samo vizualizacija nego stvaran AI/ML sustav.",
        )

    st.markdown(
        """
        <div class="note-box">
            Mala ili umjerena razlika između production i strict pristupa sugerira da model ne ovisi
            samo o jednom shortcut signalu, nego da zaista uči meteorološke i toplinske obrasce.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Tab 2: v1 and v2 ----
with tabs[1]:
    st.markdown("## Production vs Strict model analysis")

    st.markdown("### Confusion matrix")
    cm1, cm2 = st.columns(2)
    with cm1:
        fig_conf_v1 = build_confusion_heatmap(conf_v1, "Production model (v1)")
        if fig_conf_v1 is not None:
            st.plotly_chart(fig_conf_v1, use_container_width=True)
        else:
            st.info("Confusion matrix za production model nije dostupna.")
    with cm2:
        fig_conf_v2 = build_confusion_heatmap(conf_v2, "Strict model (v2)")
        if fig_conf_v2 is not None:
            st.plotly_chart(fig_conf_v2, use_container_width=True)
        else:
            st.info("Confusion matrix za strict model nije dostupna.")

    st.markdown("### Najvažniji featurei")
    fi1, fi2 = st.columns(2)
    with fi1:
        fig_feat_v1 = build_top_features_chart(feat_v1, "Top featurei — production model", top_n=15)
        if fig_feat_v1 is not None:
            st.plotly_chart(fig_feat_v1, use_container_width=True)
        else:
            st.info("Feature importance za production model nije dostupna.")
    with fi2:
        fig_feat_v2 = build_top_features_chart(feat_v2, "Top featurei — strict model", top_n=15)
        if fig_feat_v2 is not None:
            st.plotly_chart(fig_feat_v2, use_container_width=True)
        else:
            st.info("Feature importance za strict model nije dostupna.")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Top 10 featurea — production")
        if not feat_v1.empty:
            st.dataframe(feat_v1.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Top featurei nisu dostupni.")
    with t2:
        st.markdown("### Top 10 featurea — strict")
        if not feat_v2.empty:
            st.dataframe(feat_v2.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Top featurei nisu dostupni.")

    st.markdown("### Precision / recall / F1 po klasama")
    rp1, rp2 = st.columns(2)
    with rp1:
        fig_report_v1 = build_class_report_chart(report_v1, "Production model — precision / recall / F1")
        if fig_report_v1 is not None:
            st.plotly_chart(fig_report_v1, use_container_width=True)
        else:
            st.info("Classification report za production model nije dostupan.")
    with rp2:
        fig_report_v2 = build_class_report_chart(report_v2, "Strict model — precision / recall / F1")
        if fig_report_v2 is not None:
            st.plotly_chart(fig_report_v2, use_container_width=True)
        else:
            st.info("Classification report za strict model nije dostupan.")

    st.markdown("### Primjeri pogrešnih predikcija")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### Production model (v1)")
        err_v1 = build_error_examples(pred_v1)
        if not err_v1.empty:
            st.dataframe(err_v1, use_container_width=True, hide_index=True)
        else:
            st.info("Nema dostupnih pogrešnih primjera za production model.")
    with e2:
        st.markdown("#### Strict model (v2)")
        err_v2 = build_error_examples(pred_v2)
        if not err_v2.empty:
            st.dataframe(err_v2, use_container_width=True, hide_index=True)
        else:
            st.info("Nema dostupnih pogrešnih primjera za strict model.")

# ---- Tab 3: v3 ----
with tabs[2]:
    st.markdown("## Escalation model v3")

    v31, v32, v33, v34 = st.columns(4)
    with v31:
        render_metric_card(
            "Accuracy",
            f"{metrics_v3.get('accuracy', 'N/A')}",
            str(metrics_v3.get("best_model", "N/A")),
        )
    with v32:
        render_metric_card(
            "Precision positive",
            f"{metrics_v3.get('precision_positive', 'N/A')}",
            "Positive class",
        )
    with v33:
        render_metric_card(
            "Recall positive",
            f"{metrics_v3.get('recall_positive', 'N/A')}",
            "Positive class",
        )
    with v34:
        render_metric_card(
            "F1 positive",
            f"{metrics_v3.get('f1_positive', 'N/A')}",
            "Positive class",
        )

    st.markdown("### Threshold tuning")
    th1, th2 = st.columns([1.3, 1])
    with th1:
        fig_threshold = build_v3_threshold_chart(threshold_v3)
        if fig_threshold is not None:
            st.plotly_chart(fig_threshold, use_container_width=True)
        else:
            st.info("Threshold tuning graf nije dostupan.")
    with th2:
        best_by_f1 = threshold_summary_v3.get("best_by_f1", {})
        suggested_thresholds = threshold_summary_v3.get("suggested_thresholds", {})

        render_list_card(
            "Threshold tuning summary",
            [
                f"Best threshold by F1: {best_by_f1.get('threshold', 'N/A')}",
                f"Best positive F1: {best_by_f1.get('f1_positive', 'N/A')}",
                f"Best precision+: {best_by_f1.get('precision_positive', 'N/A')}",
                f"Best recall+: {best_by_f1.get('recall_positive', 'N/A')}",
                f"Stable upper bound: {suggested_thresholds.get('stable_threshold_upper', 'N/A')}",
                f"Likely escalation lower bound: {suggested_thresholds.get('likely_escalation_lower', 'N/A')}",
            ],
            "Threshold tuning je važan jer alert logika ne smije biti ni pretiha ni prealarmistična.",
        )

    st.markdown("### Confusion matrix i class summary")
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        fig_conf_v3 = build_confusion_heatmap(conf_v3, "Escalation model (v3)")
        if fig_conf_v3 is not None:
            st.plotly_chart(fig_conf_v3, use_container_width=True)
        else:
            st.info("Confusion matrix za v3 nije dostupna.")
    with c2:
        if not report_v3_df.empty:
            st.dataframe(report_v3_df, use_container_width=True, hide_index=True)
        else:
            st.info("v3 classification report nije dostupan.")

    st.markdown("### Najvažniji featurei za v3")
    fv31, fv32 = st.columns([1.2, 0.8])
    with fv31:
        fig_feat_v3 = build_top_features_chart(feat_v3, "Top featurei — escalation model v3", top_n=15)
        if fig_feat_v3 is not None:
            st.plotly_chart(fig_feat_v3, use_container_width=True)
        else:
            st.info("Feature importance za v3 nije dostupna.")
    with fv32:
        if not feat_v3.empty:
            st.dataframe(feat_v3.head(12), use_container_width=True, hide_index=True)
        else:
            st.info("Top featurei za v3 nisu dostupni.")

    st.markdown("### False positives / false negatives")
    fp_col, fn_col = st.columns(2)
    with fp_col:
        st.markdown("#### False positives")
        fp_df = build_v3_error_examples(false_pos_v3, "false_positive")
        if not fp_df.empty:
            st.dataframe(fp_df, use_container_width=True, hide_index=True)
        else:
            st.info("False positives nisu dostupni.")
    with fn_col:
        st.markdown("#### False negatives")
        fn_df = build_v3_error_examples(false_neg_v3, "false_negative")
        if not fn_df.empty:
            st.dataframe(fn_df, use_container_width=True, hide_index=True)
        else:
            st.info("False negatives nisu dostupni.")

    st.markdown(
        """
        <div class="note-box">
            v3 nije zamišljen kao zamjena za forecast, nego kao dodatni early-warning sloj koji operaterima
            i gradovima daje signal da bi se situacija mogla pogoršati prije nego što peak stvarno nastupi.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Tab 4: XAI ----
with tabs[3]:
    st.markdown("## Live XAI demo for v3 escalation signal")

    x1, x2, x3 = st.columns(3)
    with x1:
        render_metric_card(
            "XAI method",
            str(xai_summary.get("method", "N/A")),
            "Local explanation engine",
        )
    with x2:
        render_metric_card(
            "72h probability",
            f"{xai_summary.get('probability'):.2f}" if xai_summary.get("probability") is not None else "N/A",
            "Escalation probability",
        )
    with x3:
        render_metric_card(
            "Model label",
            str(xai_summary.get("label", "N/A")),
            "Current v3 signal",
        )

    st.markdown(
        f"""
        <div class="note-box">
            <b>Live XAI context:</b> grad <b>{selected_city}</b>,
            scenario mode <b>{"enabled" if xai_scenario_enabled else "disabled"}</b>,
            ΔT <b>{xai_temperature_delta:+.1f} °C</b>,
            ΔRH <b>{xai_humidity_delta:+.1f}%</b>,
            ΔWind <b>{xai_wind_delta:+.1f} m/s</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    lx1, lx2 = st.columns(2)
    with lx1:
        render_list_card(
            "Top positive drivers",
            [
                f"{item['feature']} (contribution: {item['contribution']})"
                for item in xai_summary.get("top_positive_drivers", [])
            ] or ["Nema pozitivnih drivera za prikaz."],
            "Featurei koji guraju vjerojatnost eskalacije prema gore.",
        )

    with lx2:
        render_list_card(
            "Top protective drivers",
            [
                f"{item['feature']} (contribution: {item['contribution']})"
                for item in xai_summary.get("top_protective_drivers", [])
            ] or ["Nema zaštitnih drivera za prikaz."],
            "Featurei koji guraju signal prema stabilnijem ishodu.",
        )

    st.info(xai_summary.get("explanation_text", "Nema dostupnog lokalnog objašnjenja."))

    st.markdown("### Input signal used for explanation")
    if not xai_input_row.empty:
        show_cols = [
            c for c in [
                "city",
                "date",
                "temp_max",
                "temp_min",
                "apparent_temp_max",
                "apparent_temp_mean",
                "humidity_mean",
                "wind_speed_mean",
                "heuristic_risk_level",
                "heuristic_risk_score",
                "ml_predicted_label",
                "ml_prediction_confidence",
            ]
            if c in xai_input_row.columns
        ]
        signal_df = xai_input_row[show_cols].copy()
        if "date" in signal_df.columns:
            signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.strftime("%d.%m.%Y.")
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
    else:
        st.info("Live input signal nije dostupan.")

    st.markdown(
        """
        <div class="note-box">
            Ovo je važan trust layer za projekt: sustav ne daje samo signal "Likely escalation" ili "Stable",
            nego pokušava objasniti koji su lokalni driveri najviše utjecali na odluku modela.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.success(
    """
    Zaključak: HeatSafe HR sada kombinira operativni multiclass model, strožu validacijsku verziju,
    72h escalation early-warning model i explainable AI sloj. To projektu daje i praktičnu vrijednost
    i research credibility.
    """
)