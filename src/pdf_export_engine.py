from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary, build_sector_actions
from src.escalation_engine import predict_escalation_from_features
from src.forecast_engine import make_ml_forecast
from src.impact_engine import (
    build_operational_triggers,
    identify_primary_impacts,
    identify_priority_groups,
    impact_band_from_peak,
)
from src.resource_routing_engine import (
    build_top_dispatch_summary,
    recommend_dispatch_resources,
)
from src.vulnerability_engine import (
    build_impact_adjusted_priority,
    get_city_vulnerability_snapshot,
)
from src.xai_engine import explain_escalation_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "briefings"

ACCENT = colors.HexColor("#1d4ed8")
DARK = colors.HexColor("#0f172a")
GREEN = colors.HexColor("#2E8B57")
AMBER = colors.HexColor("#E6A700")
ORANGE = colors.HexColor("#E67E22")
RED = colors.HexColor("#C0392B")
LIGHT_BG = colors.HexColor("#F8FAFC")
SOFT_BLUE = colors.HexColor("#EFF6FF")
SOFT_ORANGE = colors.HexColor("#FFF7ED")
BORDER = colors.HexColor("#D9E2EC")
MUTED = colors.HexColor("#64748B")
WHITE = colors.white

RISK_COLOR_MAP = {
    "Nizak": GREEN,
    "Umjeren": AMBER,
    "Visok": ORANGE,
    "Vrlo visok": RED,
}

READINESS_COLOR_MAP = {
    "Monitoring": GREEN,
    "Prepared": AMBER,
    "Elevated Readiness": ORANGE,
    "Critical Preparedness": RED,
}

V3_COLOR_MAP = {
    "Stable": colors.HexColor("#64748B"),
    "Watch": AMBER,
    "Likely escalation": RED,
}

CONSENSUS_COLOR_MAP = {
    "Strong consensus": GREEN,
    "Moderate consensus": AMBER,
    "Mixed signals": ORANGE,
    "Low consensus": RED,
    "Standalone mode": ACCENT,
    "Action Center mode": ACCENT,
}

CONFIDENCE_COLOR_MAP = {
    "High": GREEN,
    "Moderate": AMBER,
    "Low": RED,
    "N/A": ACCENT,
}


def _safe_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _styles() -> StyleSheet1:
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="HSH_Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_Subtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=WHITE,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=DARK,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=DARK,
            spaceBefore=3,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=DARK,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_Label",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            spaceAfter=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HSH_BoxTitle",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=DARK,
            spaceAfter=4,
        )
    )
    return styles


def _pill_table(text: str, color: colors.Color) -> Table:
    styles = _styles()
    table = Table([[Paragraph(f"<b>{_safe_text(text)}</b>", styles["HSH_Small"])]])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _kv_table(rows: list[tuple[str, str]], col_widths: list[float]) -> Table:
    styles = _styles()
    data = []

    for label, value in rows:
        data.append(
            [
                Paragraph(f"<b>{label}</b>", styles["HSH_Body"]),
                Paragraph(_safe_text(value), styles["HSH_Body"]),
            ]
        )

    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _metric_cards(metrics: list[tuple[str, str, str]], total_width: float) -> Table:
    styles = _styles()
    card_cells = []

    for label, value, sub in metrics:
        content = [
            Paragraph(label.upper(), styles["HSH_Label"]),
            Spacer(1, 2),
            Paragraph(f"<b>{_safe_text(value)}</b>", styles["HSH_H2"]),
            Paragraph(_safe_text(sub), styles["HSH_Small"]),
        ]
        inner = Table([[content]], colWidths=[(total_width / 3.0) - 8])
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                    ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        card_cells.append(inner)

    table = Table([card_cells], colWidths=[total_width / 3.0] * 3)
    table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _bullet_list(items: list[str]) -> ListFlowable:
    styles = _styles()
    return ListFlowable(
        [ListItem(Paragraph(_safe_text(item), styles["HSH_Body"])) for item in items],
        bulletType="bullet",
        leftIndent=12,
        bulletFontName="Helvetica",
        bulletFontSize=8,
    )


def _section_title(text: str):
    return Paragraph(text, _styles()["HSH_H1"])


def _subsection_title(text: str):
    return Paragraph(text, _styles()["HSH_H2"])


def _divider(width: float) -> HRFlowable:
    return HRFlowable(width=width, thickness=0.8, color=BORDER, spaceBefore=4, spaceAfter=8)


def _info_box(title: str, body: str, width: float, background: colors.Color = SOFT_BLUE) -> Table:
    styles = _styles()
    content = [
        Paragraph(f"<b>{title}</b>", styles["HSH_BoxTitle"]),
        Paragraph(body, styles["HSH_Body"]),
    ]
    table = Table([[content]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def _timeline_risk_color(level: str) -> colors.Color:
    return RISK_COLOR_MAP.get(level, LIGHT_BG)


def generate_daily_briefing_pdf(
    *,
    city: str,
    summary: dict,
    reliability_snapshot: dict | None = None,
    vulnerability_snapshot: dict | None = None,
    impact_adjusted_priority: float | None = None,
    impact_band: str | None = None,
    priority_groups: list[str] | None = None,
    primary_impacts: list[str] | None = None,
    operational_triggers: list[str] | None = None,
    sector_actions: dict | None = None,
    xai_summary: dict | None = None,
    top_dispatch_summary: str | None = None,
    public_alert_summary: str | None = None,
    timeline_df: pd.DataFrame | None = None,
    scenario_enabled: bool = False,
    scenario_meta: dict | None = None,
) -> bytes:
    styles = _styles()
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"HeatSafe HR Daily Briefing - {city}",
        author="HeatSafe HR",
    )

    usable_width = A4[0] - doc.leftMargin - doc.rightMargin
    story = []
    generated_at = datetime.now().strftime("%d.%m.%Y. %H:%M")

    hero = Table(
        [[
            [
                Paragraph("HeatSafe HR - Daily Briefing", styles["HSH_Title"]),
                Spacer(1, 4),
                Paragraph(
                    f"Operativni dnevni briefing za grad <b>{city}</b>. "
                    "Dokument spaja readiness, escalation, impact, vulnerability, XAI i dispatch routing.",
                    styles["HSH_Subtitle"],
                ),
            ]
        ]],
        colWidths=[usable_width],
    )
    hero.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), DARK),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(hero)
    story.append(Spacer(1, 10))

    meta_rows = [
        ("City", city),
        ("Generated at", generated_at),
        ("Peak date", summary["next_7d_peak_date"].strftime("%d.%m.%Y.") if summary.get("next_7d_peak_date") is not None else "N/A"),
        ("Scenario mode", "Enabled" if scenario_enabled else "Disabled"),
        (
            "Scenario details",
            ", ".join(f"{k}={v}" for k, v in (scenario_meta or {}).items()) if scenario_meta else "Baseline",
        ),
    ]
    story.append(_kv_table(meta_rows, [50 * mm, usable_width - 50 * mm]))
    story.append(Spacer(1, 10))

    story.append(
        _metric_cards(
            [
                (
                    "Readiness",
                    _safe_text(summary.get("readiness_status")),
                    _safe_text(summary.get("next_24h_level")),
                ),
                (
                    "Next 7d peak",
                    _safe_text(summary.get("next_7d_peak_level")),
                    f"{_safe_float(summary.get('next_7d_peak_score')):.1f}",
                ),
                (
                    "Impact priority",
                    f"{_safe_float(impact_adjusted_priority):.1f}" if impact_adjusted_priority is not None else "N/A",
                    _safe_text(impact_band),
                ),
            ],
            usable_width,
        )
    )
    story.append(Spacer(1, 10))

    pill_row = []
    pill_row.append(
        _pill_table(
            _safe_text(summary.get("readiness_status")),
            READINESS_COLOR_MAP.get(summary.get("readiness_status"), ACCENT),
        )
    )

    if reliability_snapshot:
        pill_row.append(
            _pill_table(
                _safe_text(reliability_snapshot.get("v3_signal")),
                V3_COLOR_MAP.get(reliability_snapshot.get("v3_signal"), ACCENT),
            )
        )
        pill_row.append(
            _pill_table(
                _safe_text(reliability_snapshot.get("confidence_level")),
                CONFIDENCE_COLOR_MAP.get(_safe_text(reliability_snapshot.get("confidence_level")), ACCENT),
            )
        )

    pill_table = Table([pill_row], hAlign="LEFT")
    pill_table.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(pill_table)
    story.append(Spacer(1, 10))

    story.append(
        _info_box(
            "Executive interpretation",
            (
                f"City <b>{city}</b> is currently assessed at readiness status "
                f"<b>{_safe_text(summary.get('readiness_status'))}</b>. "
                f"The next 7-day peak is expected to reach "
                f"<b>{_safe_text(summary.get('next_7d_peak_level'))}</b> "
                f"with score <b>{_safe_float(summary.get('next_7d_peak_score')):.1f}</b>. "
                f"This document is intended as an operator-ready briefing for planning, coordination and communication."
            ),
            usable_width,
            background=SOFT_BLUE,
        )
    )
    story.append(Spacer(1, 10))

    story.append(_section_title("1. Executive city summary"))
    story.append(_divider(usable_width))
    executive_rows = [
        ("Next 24h risk", _safe_text(summary.get("next_24h_level"))),
        ("Next 24h score", f"{_safe_float(summary.get('next_24h_score')):.1f}"),
        ("Next 72h peak", _safe_text(summary.get("next_72h_peak_level"))),
        ("Next 7d peak date", summary["next_7d_peak_date"].strftime("%d.%m.%Y.") if summary.get("next_7d_peak_date") is not None else "N/A"),
        ("High-risk days", _safe_text(summary.get("high_risk_days"))),
    ]
    story.append(_kv_table(executive_rows, [58 * mm, usable_width - 58 * mm]))
    story.append(Spacer(1, 10))

    if reliability_snapshot:
        story.append(_section_title("2. Reliability & escalation"))
        story.append(_divider(usable_width))

        reliability_rows = [
            ("v1 signal", _safe_text(reliability_snapshot.get("v1_signal"))),
            ("v2 signal", _safe_text(reliability_snapshot.get("v2_signal"))),
            ("v3 signal", _safe_text(reliability_snapshot.get("v3_signal"))),
            ("Consensus status", _safe_text(reliability_snapshot.get("consensus_status"))),
            ("Confidence level", _safe_text(reliability_snapshot.get("confidence_level"))),
            ("Reliability score", f"{_safe_float(reliability_snapshot.get('reliability_score')):.1f}"),
            ("Operator review required", "Yes" if reliability_snapshot.get("operator_review_required") else "No"),
            ("Uncertainty warning", _safe_text(reliability_snapshot.get("uncertainty_warning"))),
        ]
        story.append(_kv_table(reliability_rows, [62 * mm, usable_width - 62 * mm]))
        story.append(Spacer(1, 8))

        warning_text = _safe_text(reliability_snapshot.get("uncertainty_warning"))
        warning_bg = SOFT_ORANGE if reliability_snapshot.get("operator_review_required") else SOFT_BLUE
        story.append(_info_box("Reliability note", warning_text, usable_width, background=warning_bg))
        story.append(Spacer(1, 10))

    story.append(_section_title("3. Impact & vulnerability"))
    story.append(_divider(usable_width))

    impact_rows = [
        ("Impact band", _safe_text(impact_band)),
        ("Impact-adjusted priority", f"{_safe_float(impact_adjusted_priority):.1f}" if impact_adjusted_priority is not None else "N/A"),
        ("Vulnerability index", f"{_safe_float(vulnerability_snapshot.get('vulnerability_index')):.1f}" if vulnerability_snapshot else "N/A"),
        ("Vulnerability band", _safe_text(vulnerability_snapshot.get("vulnerability_band")) if vulnerability_snapshot else "N/A"),
    ]
    story.append(_kv_table(impact_rows, [62 * mm, usable_width - 62 * mm]))
    story.append(Spacer(1, 8))

    left_block = Table(
        [[_subsection_title("Primary impacts")], [_bullet_list(primary_impacts or ["No impact list available."])]],
        colWidths=[(usable_width / 2) - 6],
    )
    right_block = Table(
        [[_subsection_title("Priority groups")], [_bullet_list(priority_groups or ["No priority-group list available."])]],
        colWidths=[(usable_width / 2) - 6],
    )

    lr_table = Table([[left_block, right_block]], colWidths=[usable_width / 2, usable_width / 2])
    lr_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(lr_table)
    story.append(Spacer(1, 10))

    story.append(_section_title("4. XAI & operational response"))
    story.append(_divider(usable_width))

    xai_rows = [
        ("XAI method", _safe_text(xai_summary.get("method")) if xai_summary else "N/A"),
        ("v3 probability", f"{_safe_float(xai_summary.get('probability')):.2f}" if xai_summary else "N/A"),
        ("v3 label", _safe_text(xai_summary.get("label")) if xai_summary else "N/A"),
        ("Top dispatch resource", _safe_text(top_dispatch_summary)),
        ("Public alert summary", _safe_text(public_alert_summary, "No public alert summary attached.")),
    ]
    story.append(_kv_table(xai_rows, [62 * mm, usable_width - 62 * mm]))
    story.append(Spacer(1, 8))

    if xai_summary and xai_summary.get("explanation_text"):
        story.append(
            _info_box(
                "XAI summary",
                _safe_text(xai_summary.get("explanation_text")),
                usable_width,
                background=SOFT_BLUE,
            )
        )
        story.append(Spacer(1, 8))

    if operational_triggers:
        story.append(_subsection_title("Operational triggers"))
        story.append(_bullet_list(operational_triggers))
        story.append(Spacer(1, 8))

    if sector_actions:
        story.append(_subsection_title("Sector actions"))
        sector_data = [
            [
                Paragraph("<b>City</b>", styles["HSH_Body"]),
                Paragraph("<b>Public services</b>", styles["HSH_Body"]),
                Paragraph("<b>Tourism</b>", styles["HSH_Body"]),
            ],
            [
                Paragraph("<br/>".join(f"- {item}" for item in sector_actions.get("city", [])[:4]), styles["HSH_Body"]),
                Paragraph("<br/>".join(f"- {item}" for item in sector_actions.get("services", [])[:4]), styles["HSH_Body"]),
                Paragraph("<br/>".join(f"- {item}" for item in sector_actions.get("tourism", [])[:4]), styles["HSH_Body"]),
            ],
        ]
        sector_table = Table(sector_data, colWidths=[usable_width / 3.0] * 3)
        sector_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                    ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(sector_table)
        story.append(Spacer(1, 10))

    if timeline_df is not None and not timeline_df.empty:
        story.append(PageBreak())
        story.append(_section_title("5. 7-day operational timeline"))
        story.append(_divider(usable_width))

        display_df = timeline_df.copy()
        if "date" in display_df.columns:
            display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%d.%m.%Y.")

        keep_cols = [
            c for c in [
                "date",
                "heuristic_risk_level",
                "heuristic_risk_score",
                "temp_max",
                "apparent_temp_max",
                "humidity_mean",
            ]
            if c in display_df.columns
        ]
        display_df = display_df[keep_cols].head(7)

        header = [
            Paragraph("<b>Date</b>", styles["HSH_Body"]),
            Paragraph("<b>Risk level</b>", styles["HSH_Body"]),
            Paragraph("<b>Score</b>", styles["HSH_Body"]),
            Paragraph("<b>Temp max</b>", styles["HSH_Body"]),
            Paragraph("<b>App temp</b>", styles["HSH_Body"]),
            Paragraph("<b>Humidity</b>", styles["HSH_Body"]),
        ]

        table_data = [header]
        for _, row in display_df.iterrows():
            table_data.append(
                [
                    Paragraph(_safe_text(row.get("date")), styles["HSH_Body"]),
                    Paragraph(_safe_text(row.get("heuristic_risk_level")), styles["HSH_Body"]),
                    Paragraph(f"{_safe_float(row.get('heuristic_risk_score')):.1f}", styles["HSH_Body"]),
                    Paragraph(f"{_safe_float(row.get('temp_max')):.1f}", styles["HSH_Body"]),
                    Paragraph(f"{_safe_float(row.get('apparent_temp_max')):.1f}", styles["HSH_Body"]),
                    Paragraph(f"{_safe_float(row.get('humidity_mean')):.1f}", styles["HSH_Body"]),
                ]
            )

        timeline_table = Table(
            table_data,
            colWidths=[28 * mm, 34 * mm, 20 * mm, 24 * mm, 24 * mm, 24 * mm],
        )
        timeline_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                    ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )

        for row_idx in range(1, len(table_data)):
            if "heuristic_risk_level" in display_df.columns:
                level = _safe_text(display_df.iloc[row_idx - 1].get("heuristic_risk_level"))
                color = _timeline_risk_color(level)
                timeline_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (1, row_idx), (1, row_idx), color),
                            ("TEXTCOLOR", (1, row_idx), (1, row_idx), WHITE),
                        ]
                    )
                )

        story.append(timeline_table)
        story.append(Spacer(1, 10))

        story.append(
            _info_box(
                "Document note",
                "This PDF is a decision-support briefing generated by HeatSafe HR. "
                "It is intended to support human operators, local authorities and planners during elevated heat-risk conditions.",
                usable_width,
                background=SOFT_BLUE,
            )
        )

    def _add_page_elements(canvas, doc_):
        canvas.saveState()

        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(DARK)
        canvas.drawString(doc_.leftMargin, A4[1] - 9 * mm, "HeatSafe HR - Daily Briefing")

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - doc_.rightMargin, A4[1] - 9 * mm, city)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc_.leftMargin, 8 * mm, f"Generated: {generated_at}")
        canvas.drawRightString(A4[0] - doc_.rightMargin, 8 * mm, f"Page {doc_.page}")

        canvas.restoreState()

    doc.build(story, onFirstPage=_add_page_elements, onLaterPages=_add_page_elements)
    buffer.seek(0)
    return buffer.getvalue()


def save_daily_briefing_pdf(output_path: Path, **kwargs) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = generate_daily_briefing_pdf(**kwargs)
    output_path.write_bytes(pdf_bytes)
    return output_path


if __name__ == "__main__":
    city = DEFAULT_CITY

    forecast_df = make_ml_forecast(city)
    if "city" not in forecast_df.columns:
        forecast_df["city"] = city

    summary = build_city_readiness_summary(city, forecast_df)
    sector_actions = build_sector_actions(summary["next_7d_peak_level"])

    first_row_df = forecast_df.sort_values("date").head(1).copy()
    if "city" not in first_row_df.columns:
        first_row_df["city"] = city

    escalation_df = predict_escalation_from_features(first_row_df)
    escalation_row = escalation_df.iloc[0]

    vulnerability_snapshot = get_city_vulnerability_snapshot(city)
    impact_adjusted_priority = build_impact_adjusted_priority(
        next_7d_peak_score=float(summary["next_7d_peak_score"]),
        escalation_probability_72h=float(escalation_row["escalation_probability_72h"]),
        vulnerability_index=float(vulnerability_snapshot["vulnerability_index"]),
    )

    priority_groups = identify_priority_groups(summary, str(escalation_row["escalation_label_72h"]))
    primary_impacts = identify_primary_impacts(summary, str(escalation_row["escalation_label_72h"]))
    operational_triggers = build_operational_triggers(summary, str(escalation_row["escalation_label_72h"]))
    impact_band = impact_band_from_peak(summary["next_7d_peak_level"], str(escalation_row["escalation_label_72h"]))

    dispatch_df = recommend_dispatch_resources(
        city=city,
        escalation_label=str(escalation_row["escalation_label_72h"]),
        priority_groups=priority_groups,
        top_n=5,
    )
    top_dispatch_summary = build_top_dispatch_summary(dispatch_df)

    xai_summary = explain_escalation_row(first_row_df)

    reliability_snapshot = {
        "v1_signal": str(first_row_df.iloc[0].get("ml_predicted_label", "N/A")),
        "v2_signal": "N/A",
        "v3_signal": str(escalation_row["escalation_label_72h"]),
        "confidence_level": "N/A",
        "reliability_score": 0,
        "operator_review_required": False,
        "uncertainty_warning": "Standalone PDF test mode.",
        "consensus_status": "Standalone mode",
    }

    output_path = OUTPUT_DIR / f"heatsafe_hr_daily_briefing_{city}.pdf"
    save_daily_briefing_pdf(
        output_path=output_path,
        city=city,
        summary=summary,
        reliability_snapshot=reliability_snapshot,
        vulnerability_snapshot=vulnerability_snapshot,
        impact_adjusted_priority=impact_adjusted_priority,
        impact_band=impact_band,
        priority_groups=priority_groups,
        primary_impacts=primary_impacts,
        operational_triggers=operational_triggers,
        sector_actions=sector_actions,
        xai_summary=xai_summary,
        top_dispatch_summary=top_dispatch_summary,
        public_alert_summary="Public advisory package can be attached from Alert Center communication layer.",
        timeline_df=forecast_df,
        scenario_enabled=False,
        scenario_meta=None,
    )
    print(f"[OK] Saved PDF to: {output_path}")