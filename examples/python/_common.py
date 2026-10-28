"""Shared setup for the examples, which makes the tiopt package importable from the repository and picks a folder for
figures.
"""

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "examples"
OUTPUT = EXAMPLES / "output"

sys.path.insert(0, str(REPOSITORY / "python"))

import matplotlib  # noqa: E402

# Save figures to files rather than opening windows, so that the examples also run without a display
matplotlib.use("Agg")


def save(figure, name):
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / name
    figure.savefig(path, dpi=150)
    print(f"Saved {path.relative_to(REPOSITORY)}")
