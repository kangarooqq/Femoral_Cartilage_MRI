
"""Shared paths for the SKM-TEA analysis scripts.

No Stanford source data are shipped with this repository. Configure their
location using the SKMTEA_DATA_ROOT environment variable before running.
"""

import os
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parent

# Default private data directory next to the scripts (never commit its contents).
DATA_ROOT = Path(os.environ.get("SKMTEA_DATA_ROOT", REPOSITORY_DIR / "data")).expanduser().resolve()

# Output folder may be overridden independently if required.
ANALYSIS_DIR = Path(os.environ.get("SKMTEA_ANALYSIS_DIR", DATA_ROOT / "analysis_20260927")).expanduser().resolve()

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
