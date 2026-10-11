"""Shared paths for the SKM-TEA analysis scripts.

No Stanford source data are shipped with this repository. Configure their
location using the SKMTEA_DATA_ROOT environment variable before running.
"""

"""
Project path configuration.

Please modify these paths according to your local environment.
"""

from pathlib import Path

PROJECT_ROOT = Path("/your/project/root")

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
