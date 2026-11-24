"""Run steps 2 to 4 (panel, backtest, figures and tables), then build the paper if LaTeX is installed.

Step 1, the WRDS download, is run on its own: pipeline/01_download_wrds.py.

    python run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["pipeline/02_build_panel.py", "pipeline/03_backtest.py", "pipeline/04_figures_tables.py"]
PAPER = "Bajwa_PB_ROE_Pakistan.tex"

for step in STEPS:
    print("===", step, flush=True)
    if subprocess.run([sys.executable, step], cwd=ROOT).returncode != 0:
        sys.exit(f"FAILED at {step}")

try:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", PAPER], cwd=ROOT / "paper", check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    print(f"LaTeX not available or failed: compile paper/{PAPER} yourself (for example on Overleaf)")
print("=== ALL DONE")
