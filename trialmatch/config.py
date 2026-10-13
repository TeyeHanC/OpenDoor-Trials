"""Runtime settings, read once from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader: KEY=VALUE lines, existing environment wins."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    data_dir: Path
    # Notes at or below this length are sent to the model whole, so a retrieval
    # miss can never be the reason a criterion is judged "unknown".
    full_note_max_chars: int = 6000
    top_k_segments: int = 6
    segment_max_chars: int = 400
    judge_batch_size: int = 8
    judge_workers: int = 4

    @property
    def trials_dir(self) -> Path:
        return self.data_dir / "cache" / "trials"

    @property
    def decomposition_dir(self) -> Path:
        return self.data_dir / "cache" / "decompositions"

    @property
    def trec_dir(self) -> Path:
        return self.data_dir / "trec"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def reviews_path(self) -> Path:
        return self.data_dir / "reviews.jsonl"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


def get_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.environ.get("TRIALMATCH_DATA_DIR", REPO_ROOT / "data"))
    return Settings(
        model=os.environ.get("TRIALMATCH_MODEL", "claude-opus-5-5"),
        effort=os.environ.get("TRIALMATCH_EFFORT", "high"),
        data_dir=data_dir,
    )
