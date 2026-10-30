"""Put ``../src`` on ``sys.path`` so the app runs without installing the package (deploy).

Every page imports this module first: ``import _bootstrap  # noqa: F401``.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
