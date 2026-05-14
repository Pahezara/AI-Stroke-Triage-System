from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunManager:
    run_id: str
    run_dir: Path

    @staticmethod
    def create(base_dir: str = "runs", prefix: str = "run") -> "RunManager":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{ts}"
        run_dir = Path(base_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        return RunManager(run_id, run_dir)

    def save_json(self, name: str, obj: Dict[str, Any]) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def save_text(self, name: str, text: str) -> None:
        path = self.run_dir / name
        path.write_text(text, encoding="utf-8")
