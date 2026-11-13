
"""Keep test scratch files inside the workspace on managed Windows machines."""
from pathlib import Path
import tempfile
import sys
_dependencies=Path(__file__).resolve().parents[1]/".testdeps"
if _dependencies.exists():
    sys.path.insert(0,str(_dependencies))
_scratch=Path(__file__).resolve().parents[1]/"artifacts"/"test-tmp"
_scratch.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(_scratch)
