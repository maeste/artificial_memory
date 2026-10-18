"""Package storage — proprietà esclusiva dell'esecutore storage (vedi untracked_doc/).

API pubblica del modulo (contratti §4): BrainStorage. CLI: python -m src.storage.
"""

from src.storage.db import BrainStorage

__all__ = ["BrainStorage"]
