
#!/usr/bin/env python3
"""Rebuild the local Chroma index from approved canonical records."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task2.src.vector_memory import rebuild_index


if __name__ == "__main__":
    count = rebuild_index()
    print(f"Chroma index ready: {count} records.")
