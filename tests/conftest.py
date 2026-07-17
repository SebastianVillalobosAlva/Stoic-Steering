import sys
from pathlib import Path

# Make `stoic` importable when running pytest from a checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
