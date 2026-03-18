"""
Structured audit logging for Shona CSM training runs.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught  # pragma: no cover
        return {
            "command": command,
            "error": str(exc),
        }


def collect_environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": utc_now_iso(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torchaudio_version": None,
        "wandb_api_key_present": bool(os.getenv("WANDB_API_KEY")),
        "csm_repo_path": os.getenv("CSM_REPO_PATH"),
    }

    try:
        import torchaudio  # type: ignore[import-untyped]

        snapshot["torchaudio_version"] = torchaudio.__version__
    except Exception as exc:  # pylint: disable=broad-exception-caught  # pragma: no cover
        snapshot["torchaudio_version_error"] = str(exc)

    if torch.cuda.is_available():
        snapshot["cuda_device_name"] = torch.cuda.get_device_name(0)
        snapshot["cuda_capability"] = list(torch.cuda.get_device_capability(0))

    snapshot["commands"] = {
        "pip_freeze": _run_command(["python", "-m", "pip", "freeze"]),
        "ffmpeg_version": _run_command(["ffmpeg", "-version"]),
        "nvidia_smi": _run_command(["nvidia-smi"]),
    }
    return snapshot


@dataclass
class AuditLogger:
    run_dir: Path
    manifest: dict[str, Any]
    audit_dir: Path = field(init=False)
    events_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    summary_path: Path = field(init=False)
    summary: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.audit_dir = self.run_dir / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.audit_dir / "events.jsonl"
        self.manifest_path = self.audit_dir / "run_manifest.json"
        self.summary_path = self.audit_dir / "run_summary.json"
        self.summary = {
            "started_at": utc_now_iso(),
            "status": "running",
            "best_val_loss": None,
            "final_step": 0,
            "final_epoch": 0,
            "total_events": 0,
        }
        self.write_json(self.manifest_path, self.manifest)
        self.write_json(self.summary_path, self.summary)

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def log_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.summary["total_events"] += 1

    def update_summary(self, **updates: Any) -> None:
        self.summary.update(updates)
        self.write_json(self.summary_path, self.summary)

