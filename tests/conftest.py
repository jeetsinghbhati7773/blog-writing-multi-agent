from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root (which contains the ``src`` package) is importable
# regardless of the directory pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the test suite hermetic and offline: use the deterministic lightweight
# hash embeddings instead of downloading the ~90MB all-MiniLM-L6-v2 model.
# (Set before any test imports the RAG stack, which reads this at call time.)
os.environ.setdefault("EMBEDDING_PROVIDER", "lightweight")
