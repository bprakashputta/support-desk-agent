"""Settings from environment variables (and `.env` for local runs)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    agent_model: str
    user_model: str
    spend_cap_usd: float
    approval_threshold: float
    guard_mode: str
    data_dir: Path
    tau_dir: Path
    kb_dir: Path
    db_path: Path
    checkpoint_path: Path
    log_level: str

    @classmethod
    def load(cls) -> Settings:
        _load_dotenv(ROOT / ".env")
        data = Path(os.environ.get("DATA_DIR", ROOT / "data"))
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            agent_model=os.environ.get("AGENT_MODEL", "claude-haiku-5-5"),
            user_model=os.environ.get("USER_SIM_MODEL", "claude-haiku-5-5"),
            spend_cap_usd=float(os.environ.get("SPEND_CAP_USD", "5.00")),
            approval_threshold=float(os.environ.get("APPROVAL_THRESHOLD_USD", "1000")),
            guard_mode=os.environ.get("GUARD_MODE", "enforce"),
            data_dir=data,
            tau_dir=Path(os.environ.get("TAU_DIR", data / "tau_bench")),
            kb_dir=Path(os.environ.get("KB_DIR", ROOT / "kb")),
            db_path=Path(os.environ.get("DB_PATH", data / "support_desk.sqlite")),
            checkpoint_path=Path(os.environ.get("CHECKPOINT_PATH", data / "checkpoints.sqlite")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
