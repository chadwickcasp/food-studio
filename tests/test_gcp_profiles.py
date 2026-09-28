from __future__ import annotations

import subprocess
from pathlib import Path


def test_gcp_profiles() -> None:
    script = Path(__file__).resolve().parent / "test_gcp_profiles.sh"
    subprocess.run(["bash", str(script)], check=True)
