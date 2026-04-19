from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATUS_DIR = PROJECT_ROOT / "data" / "system"
STATUS_PATH = STATUS_DIR / "last_data_refresh.json"

CITIES = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]


def run_step(step_name: str, command: list[str]) -> dict[str, Any]:
    started_at = datetime.now()

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    finished_at = datetime.now()
    duration_seconds = round((finished_at - started_at).total_seconds(), 2)

    return {
        "step_name": step_name,
        "command": " ".join(command),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "success": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def save_refresh_status(status: dict[str, Any]) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def load_refresh_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {}
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def refresh_operational_data(
    cities: list[str] | None = None,
    rebuild_escalation_dataset: bool = True,
    retrain_models: bool = False,
    fail_fast: bool = True,
) -> dict[str, Any]:
    cities = cities or CITIES
    steps: list[dict[str, Any]] = []

    overall_started_at = datetime.now()

    # 1) Osvježi raw podatke po gradu
    for city in cities:
        step = run_step(
            step_name=f"data_ingestion::{city}",
            command=[sys.executable, "-m", "src.data_ingestion", "--city", city],
        )
        steps.append(step)

        if fail_fast and not step["success"]:
            status = {
                "success": False,
                "message": f"Refresh stopped on city step: {city}",
                "started_at": overall_started_at.isoformat(),
                "finished_at": datetime.now().isoformat(),
                "cities": cities,
                "rebuild_escalation_dataset": rebuild_escalation_dataset,
                "retrain_models": retrain_models,
                "steps": steps,
            }
            save_refresh_status(status)
            return status

    # 2) Rebuild processed datasets
    preprocessing_step = run_step(
        step_name="preprocessing",
        command=[sys.executable, "-m", "src.preprocessing"],
    )
    steps.append(preprocessing_step)

    if fail_fast and not preprocessing_step["success"]:
        status = {
            "success": False,
            "message": "Refresh stopped on preprocessing step.",
            "started_at": overall_started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "cities": cities,
            "rebuild_escalation_dataset": rebuild_escalation_dataset,
            "retrain_models": retrain_models,
            "steps": steps,
        }
        save_refresh_status(status)
        return status

    # 3) Rebuild escalation dataset
    if rebuild_escalation_dataset:
        escalation_step = run_step(
            step_name="build_escalation_dataset",
            command=[sys.executable, "-m", "src.build_escalation_dataset"],
        )
        steps.append(escalation_step)

        if fail_fast and not escalation_step["success"]:
            status = {
                "success": False,
                "message": "Refresh stopped on escalation dataset step.",
                "started_at": overall_started_at.isoformat(),
                "finished_at": datetime.now().isoformat(),
                "cities": cities,
                "rebuild_escalation_dataset": rebuild_escalation_dataset,
                "retrain_models": retrain_models,
                "steps": steps,
            }
            save_refresh_status(status)
            return status

    # 4) Retrain models samo ako izričito želiš
    if retrain_models:
        model_steps = [
            ("train_model", [sys.executable, "-m", "src.train_model"]),
            ("train_model_strict", [sys.executable, "-m", "src.train_model_strict"]),
            ("train_escalation_model", [sys.executable, "-m", "src.train_escalation_model"]),
        ]

        for step_name, command in model_steps:
            step = run_step(step_name=step_name, command=command)
            steps.append(step)

            if fail_fast and not step["success"]:
                status = {
                    "success": False,
                    "message": f"Refresh stopped on model step: {step_name}",
                    "started_at": overall_started_at.isoformat(),
                    "finished_at": datetime.now().isoformat(),
                    "cities": cities,
                    "rebuild_escalation_dataset": rebuild_escalation_dataset,
                    "retrain_models": retrain_models,
                    "steps": steps,
                }
                save_refresh_status(status)
                return status

    overall_finished_at = datetime.now()
    success = all(step["success"] for step in steps)

    status = {
        "success": success,
        "message": "Live data refresh completed successfully." if success else "Live data refresh completed with errors.",
        "started_at": overall_started_at.isoformat(),
        "finished_at": overall_finished_at.isoformat(),
        "duration_seconds": round((overall_finished_at - overall_started_at).total_seconds(), 2),
        "cities": cities,
        "rebuild_escalation_dataset": rebuild_escalation_dataset,
        "retrain_models": retrain_models,
        "steps": steps,
    }

    save_refresh_status(status)
    return status


def format_refresh_report(status: dict[str, Any]) -> str:
    if not status:
        return "No refresh status available."

    lines = [
        "HEATSAFE HR — LIVE DATA REFRESH REPORT",
        f"Success: {status.get('success')}",
        f"Message: {status.get('message')}",
        f"Started at: {status.get('started_at')}",
        f"Finished at: {status.get('finished_at')}",
        f"Duration (s): {status.get('duration_seconds', 'N/A')}",
        "",
        "STEPS:",
    ]

    for step in status.get("steps", []):
        lines.extend(
            [
                f"- {step['step_name']}",
                f"  Success: {step['success']}",
                f"  Duration: {step['duration_seconds']} s",
                f"  Command: {step['command']}",
                f"  Return code: {step['return_code']}",
            ]
        )
        if step.get("stdout"):
            lines.append("  STDOUT:")
            lines.append(f"  {step['stdout']}")
        if step.get("stderr"):
            lines.append("  STDERR:")
            lines.append(f"  {step['stderr']}")
        lines.append("")

    return "\n".join(lines).strip()


def main() -> None:
    status = refresh_operational_data(
        rebuild_escalation_dataset=True,
        retrain_models=False,
        fail_fast=True,
    )

    print(format_refresh_report(status))

    if not status["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()