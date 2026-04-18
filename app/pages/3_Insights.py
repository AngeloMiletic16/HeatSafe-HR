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


OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

MODEL_ANALYSIS_DIR = OUTPUTS_DIR / "model_analysis"
MODEL_ANALYSIS_STRICT_DIR = OUTPUTS_DIR / "model_analysis_strict"

METRICS_V1_PATH = MODELS_DIR / "model_metrics.json"
METRICS_V2_PATH = MODELS_DIR / "model_metrics_strict.json"

CONFUSION_V1_PATH = MODEL_ANALYSIS_DIR / "confusion_matrix.csv"
CONFUSION_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "confusion_matrix_strict.csv"

FEATURES_V1_PATH = MODEL_ANALYSIS_DIR / "feature_importance.csv"
FEATURES_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "feature_importance_strict.csv"

REPORT_V1_PATH = MODEL_ANALYSIS_DIR / "classification_report.csv"
REPORT_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "classification_report_strict.csv"

PRED_V1_PATH = MODEL_ANALYSIS_DIR / "test_predictions_detailed.csv"
PRED_V2_PATH = MODEL_ANALYSIS_STRICT_DIR / "test_predictions_detailed_strict.csv"


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


def extract_best_metrics(metrics_json: dict) -> dict:
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
    return pd.DataFrame(rows)


def style_confusion_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df


def build_top_features_chart(df: pd.DataFrame, title: str, top_n: int = 15):
    if df.empty:
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
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_error_examples(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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
    if report_df.empty:
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


st.title("🧠 Insights")

st.markdown(
    """
    Ova stranica prikazuje **AI/ML srce** sustava HeatSafe HR:
    usporedbu modela, confusion matrix, najvažnije featuree i tipične pogreške modela.
    """
)

# Load everything
metrics_v1_raw = load_json(METRICS_V1_PATH)
metrics_v2_raw = load_json(METRICS_V2_PATH)

metrics_v1 = extract_best_metrics(metrics_v1_raw)
metrics_v2 = extract_best_metrics(metrics_v2_raw)

comparison_df = build_model_comparison_df(metrics_v1, metrics_v2)

conf_v1 = load_csv(CONFUSION_V1_PATH, index_col=0)
conf_v2 = load_csv(CONFUSION_V2_PATH, index_col=0)

feat_v1 = load_csv(FEATURES_V1_PATH)
feat_v2 = load_csv(FEATURES_V2_PATH)

report_v1 = load_csv(REPORT_V1_PATH)
report_v2 = load_csv(REPORT_V2_PATH)

pred_v1 = load_csv(PRED_V1_PATH)
pred_v2 = load_csv(PRED_V2_PATH)

# --- Model summary ---
st.markdown("## Usporedba modela")

if not comparison_df.empty:
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
else:
    st.warning("Model metrics nisu dostupne.")

summary_cols = st.columns(4)
summary_cols[0].metric("Production Macro F1", f"{metrics_v1.get('macro_f1', 'N/A')}")
summary_cols[1].metric("Strict Macro F1", f"{metrics_v2.get('macro_f1', 'N/A')}")
summary_cols[2].metric("Production Accuracy", f"{metrics_v1.get('accuracy', 'N/A')}")
summary_cols[3].metric("Strict Accuracy", f"{metrics_v2.get('accuracy', 'N/A')}")

st.markdown(
    """
    **Production model (v1)** koristi širi skup signala za operativni rad platforme.  
    **Strict model (v2)** služi kao metodološki stroža validacijska verzija koja ne koristi
    `heat_risk_score*` featuree. Mala razlika u performansi između ta dva pristupa
    pokazuje da sustav ima stvarnu prediktivnu vrijednost na temelju meteoroloških obrazaca.
    """
)

st.divider()

# --- Confusion matrices ---
st.markdown("## Confusion matrix")

cm_col_1, cm_col_2 = st.columns(2)

with cm_col_1:
    st.markdown("### Production model (v1)")
    if not conf_v1.empty:
        st.dataframe(conf_v1, use_container_width=True)
    else:
        st.info("Confusion matrix za production model nije dostupna.")

with cm_col_2:
    st.markdown("### Strict model (v2)")
    if not conf_v2.empty:
        st.dataframe(conf_v2, use_container_width=True)
    else:
        st.info("Confusion matrix za strict model nije dostupna.")

st.markdown(
    """
    Confusion matrix pomaže u razumijevanju gdje model griješi.  
    Najčešće pogreške očekivano nastaju između **susjednih razina rizika**
    (npr. Umjeren ↔ Visok), što je za ovakav problem prirodno.
    """
)

st.divider()

# --- Feature importance ---
st.markdown("## Najvažniji featurei")

fi_col_1, fi_col_2 = st.columns(2)

with fi_col_1:
    st.markdown("### Production model (v1)")
    fig_v1 = build_top_features_chart(feat_v1, "Top featurei — production model", top_n=15)
    if fig_v1 is not None:
        st.plotly_chart(fig_v1, use_container_width=True)
    else:
        st.info("Feature importance za production model nije dostupna.")

with fi_col_2:
    st.markdown("### Strict model (v2)")
    fig_v2 = build_top_features_chart(feat_v2, "Top featurei — strict model", top_n=15)
    if fig_v2 is not None:
        st.plotly_chart(fig_v2, use_container_width=True)
    else:
        st.info("Feature importance za strict model nije dostupna.")

top_col_1, top_col_2 = st.columns(2)

with top_col_1:
    st.markdown("### Top 10 featurea — production")
    if not feat_v1.empty:
        st.dataframe(feat_v1.head(10), use_container_width=True, hide_index=True)

with top_col_2:
    st.markdown("### Top 10 featurea — strict")
    if not feat_v2.empty:
        st.dataframe(feat_v2.head(10), use_container_width=True, hide_index=True)

st.divider()

# --- Classification report ---
st.markdown("## Preciznost po klasama")

report_col_1, report_col_2 = st.columns(2)

with report_col_1:
    fig_report_v1 = build_class_report_chart(report_v1, "Production model — precision / recall / F1")
    if fig_report_v1 is not None:
        st.plotly_chart(fig_report_v1, use_container_width=True)
    else:
        st.info("Classification report za production model nije dostupan.")

with report_col_2:
    fig_report_v2 = build_class_report_chart(report_v2, "Strict model — precision / recall / F1")
    if fig_report_v2 is not None:
        st.plotly_chart(fig_report_v2, use_container_width=True)
    else:
        st.info("Classification report za strict model nije dostupan.")

st.divider()

# --- Error examples ---
st.markdown("## Primjeri pogrešnih predikcija")

err_col_1, err_col_2 = st.columns(2)

with err_col_1:
    st.markdown("### Production model (v1)")
    err_v1 = build_error_examples(pred_v1)
    if not err_v1.empty:
        st.dataframe(err_v1, use_container_width=True, hide_index=True)
    else:
        st.info("Nema dostupnih pogrešnih primjera ili file nije dostupan.")

with err_col_2:
    st.markdown("### Strict model (v2)")
    err_v2 = build_error_examples(pred_v2)
    if not err_v2.empty:
        st.dataframe(err_v2, use_container_width=True, hide_index=True)
    else:
        st.info("Nema dostupnih pogrešnih primjera ili file nije dostupan.")

st.divider()

# --- Interpretation section ---
st.markdown("## Tumačenje modela")

st.write(
    """
    Trenutna analiza pokazuje da modeli HeatSafe HR najveću važnost pridaju:
    - apparent temperaturi
    - maksimalnoj i prosječnoj temperaturi
    - kratkoročnim toplinskim obrascima kroz rolling featuree
    - signalima koji opisuju trajanje toplinskog opterećenja

    To je u skladu s očekivanom fizikalnom logikom toplinskog stresa i daje projektu
    dodatnu vjerodostojnost.
    """
)

st.success(
    """
    Zaključak: HeatSafe HR ne funkcionira samo kao dashboard, nego kao stvarni
    AI/ML sustav za procjenu toplinskog rizika, s operativnim i strogo validacijskim modelom.
    """
)