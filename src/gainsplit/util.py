"""Small helpers with no torch dependency."""
from pathlib import Path
import subprocess


def git_commit() -> str:
    """Commit of the code that produced a run, so every result can be traced."""
    try:
        repo = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
