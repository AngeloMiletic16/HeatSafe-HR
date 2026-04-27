from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STATUS_DIR = DATA_DIR / "status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

REFRESH_STATUS_PATH = STATUS_DIR / "refresh_status.json"
REFRESH_AUDIT_PATH = STATUS_DIR / "refresh_audit_log.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_refresh_status() -> dict:
    return _read_json(
        REFRESH_STATUS_PATH,
        {
            "last_refresh_utc": None,
            "status": "never_run",
            "message": "Refresh još nije pokrenut.",
            "cities_updated": 0,
            "errors": [],
        },
    )


def load_refresh_audit_log(limit: int = 5) -> list[dict]:
    log = _read_json(REFRESH_AUDIT_PATH, [])
    if not isinstance(log, list):
        return []
    return log[:limit]


def append_refresh_audit(entry: dict, keep: int = 20) -> None:
    log = _read_json(REFRESH_AUDIT_PATH, [])
    if not isinstance(log, list):
        log = []

    log.insert(0, entry)
    log = log[:keep]
    _write_json(REFRESH_AUDIT_PATH, log)


def get_data_freshness_info(stale_after_hours: float = 3.0, warning_after_hours: float = 1.5) -> dict:
    status = load_refresh_status()
    last_refresh = status.get("last_refresh_utc")

    if not last_refresh:
        return {
            "badge_label": "No data refresh",
            "badge_color": "#64748b",
            "freshness_state": "unknown",
            "age_hours": None,
            "message": "Nema zabilježenog uspješnog refresha podataka.",
        }

    try:
        last_dt = datetime.fromisoformat(last_refresh)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now_dt = datetime.now(timezone.utc)
        age_hours = (now_dt - last_dt).total_seconds() / 3600.0
    except Exception:
        return {
            "badge_label": "Refresh parse error",
            "badge_color": "#C0392B",
            "freshness_state": "error",
            "age_hours": None,
            "message": "Vrijeme zadnjeg refresha nije moguće pročitati.",
        }

    if age_hours <= warning_after_hours:
        return {
            "badge_label": "Fresh data",
            "badge_color": "#2E8B57",
            "freshness_state": "fresh",
            "age_hours": age_hours,
            "message": f"Podaci su svježi. Zadnji refresh bio je prije približno {age_hours:.1f} h.",
        }

    if age_hours <= stale_after_hours:
        return {
            "badge_label": "Aging data",
            "badge_color": "#E6A700",
            "freshness_state": "warning",
            "age_hours": age_hours,
            "message": f"Podaci stare. Zadnji refresh bio je prije približno {age_hours:.1f} h.",
        }

    return {
        "badge_label": "Stale data",
        "badge_color": "#C0392B",
        "freshness_state": "stale",
        "age_hours": age_hours,
        "message": f"Podaci su zastarjeli. Zadnji refresh bio je prije približno {age_hours:.1f} h.",
    }


def refresh_operational_data() -> dict:
    """
    Placeholder refresh orchestrator.

    Ovdje možeš kasnije spojiti:
    - live weather pull
    - preprocessing
    - risk engine
    - forecast refresh
    - alert refresh

    Za sada upisuje status i audit trail tako da app ima stvarni operational refresh layer.
    """
    started_at = _utc_now_iso()

    try:
        # Placeholder: ovdje kasnije možeš zvati pravi data pipeline
        refreshed_cities = [
            "Dubrovnik",
            "Osijek",
            "Rijeka",
            "Split",
            "Šibenik",
            "Zadar",
            "Zagreb",
        ]

        status_payload = {
            "last_refresh_utc": started_at,
            "status": "success",
            "message": "Operational data refresh completed successfully.",
            "cities_updated": len(refreshed_cities),
            "cities": refreshed_cities,
            "errors": [],
        }
        _write_json(REFRESH_STATUS_PATH, status_payload)

        append_refresh_audit(
            {
                "timestamp_utc": started_at,
                "status": "success",
                "message": "Operational data refresh completed successfully.",
                "cities_updated": len(refreshed_cities),
                "errors_count": 0,
            }
        )
        return status_payload

    except Exception as exc:
        error_payload = {
            "last_refresh_utc": started_at,
            "status": "failed",
            "message": f"Operational data refresh failed: {exc}",
            "cities_updated": 0,
            "errors": [str(exc)],
        }
        _write_json(REFRESH_STATUS_PATH, error_payload)

        append_refresh_audit(
            {
                "timestamp_utc": started_at,
                "status": "failed",
                "message": str(exc),
                "cities_updated": 0,
                "errors_count": 1,
            }
        )
        return error_payload