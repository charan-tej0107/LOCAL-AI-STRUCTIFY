"""Application entry point — bootstrap and launcher.

Usage:

    python app.py              # validate environment & launch Streamlit UI
    python app.py --check      # validate environment only & exit
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from utils.logger import setup_logging, get_logger
from utils.system_check import run_all_checks


def run_health_check() -> int:
    """Validate the environment and return an exit code.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    setup_logging()
    log = get_logger("app")

    log.info("=" * 60)
    log.info("  Local AI Structify — environment check")
    log.info("=" * 60)

    health = run_all_checks()

    if not health.all_passed:
        log.error("Environment validation FAILED — see messages above.")
        log.error("Fix the reported issues and re-run the application.")
        return 1

    log.info("All system checks passed — environment is ready.")
    return 0


def launch_ui() -> int:
    """Run health checks then launch the Streamlit UI.

    Returns:
        Exit code of the Streamlit process.
    """
    code = run_health_check()
    if code != 0:
        return code

    log = get_logger("app")
    ui_entry = Path(__file__).resolve().parent / "ui" / "app.py"

    if not ui_entry.is_file():
        log.error("UI entry point not found: %s", ui_entry)
        return 1

    log.info("Launching Streamlit UI …")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_entry),
        "--browser.serverAddress=localhost",
        "--server.headless=true",
    ]
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        log.info("Shutting down.")
        return 0
    except Exception as exc:
        log.exception("Failed to launch Streamlit: %s", exc)
        return 1


def main() -> int:
    """CLI entry point."""
    if "--check" in sys.argv:
        return run_health_check()
    return launch_ui()


if __name__ == "__main__":
    sys.exit(main())
