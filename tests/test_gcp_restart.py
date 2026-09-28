from __future__ import annotations

import subprocess
from pathlib import Path


def test_gcp_restart_decisions() -> None:
    script = Path(__file__).resolve().parent / "test_gcp_restart.sh"
    subprocess.run(["bash", str(script)], check=True)
