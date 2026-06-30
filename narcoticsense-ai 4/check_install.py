from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))
from narcoticsense.feature_engineering import peak_table
from narcoticsense.preprocessing import PreprocessingPipeline
from narcoticsense.spectroscopy import Spectrum
print("OK: NarcoticSense imports work")
