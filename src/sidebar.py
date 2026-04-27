from __future__ import annotations

import streamlit as st


def render_app_sidebar(
    *,
    selected_city: str | None = None,
    risk_level: str | None = None,
    readiness_status: str | None = None,
    escalation_label: str | None = None,
    escalation_probability: float | None = None,
) -> None:
    st.sidebar.title("HeatSafe HR")

    st.sidebar.markdown("### Status")
    if selected_city is not None:
        st.sidebar.markdown(f"**Grad:** {selected_city}")
    if risk_level is not None:
        st.sidebar.markdown(f"**Risk level:** {risk_level}")
    if readiness_status is not None:
        st.sidebar.markdown(f"**Readiness:** {readiness_status}")
    if escalation_label is not None and escalation_probability is not None:
        st.sidebar.markdown(
            f"**72h escalation:** {escalation_label} ({escalation_probability:.2f})"
        )
    elif escalation_label is not None:
        st.sidebar.markdown(f"**72h escalation:** {escalation_label}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🟢 OPERATOR FLOW")
    st.sidebar.page_link("Home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/1_Overview.py", label="Overview", icon="📊")
    st.sidebar.page_link("pages/6_Command_Dashboard.py", label="Command Dashboard", icon="🧭")
    st.sidebar.page_link("pages/5_Action_Center.py", label="Action Center", icon="🚨")
    st.sidebar.page_link("pages/10_Alert_Center.py", label="Alert Center", icon="📢")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 ANALYSIS")
    st.sidebar.page_link("pages/3_Insights.py", label="Insights", icon="🧠")
    st.sidebar.page_link("pages/4_Forecast.py", label="Forecast", icon="🔮")
    st.sidebar.page_link("pages/2_History.py", label="History", icon="🕘")
    st.sidebar.page_link("pages/11_Historical_Replay.py", label="Historical Replay", icon="⏪")
    st.sidebar.page_link("pages/12_Stress_Test.py", label="Stress Test", icon="🔥")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🌍 PUBLIC & INFO")
    st.sidebar.page_link("pages/8_Public_Advisory.py", label="Public Advisory", icon="📣")
    st.sidebar.page_link("pages/9_Resources_Map.py", label="Resources Map", icon="🧊")
    st.sidebar.page_link("pages/7_Methodology_Research.py", label="Methodology / Research", icon="🧪")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "HeatSafe HR je AI/ML decision-support platforma za toplinski rizik, "
        "readiness, vulnerability-aware prioritization i operativnu komunikaciju."
    )