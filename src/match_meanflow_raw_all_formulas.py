"""Old name of src/uflowcsp.py, kept so existing commands and scripts keep working."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("uflowcsp.py")), run_name="__main__")
