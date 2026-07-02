import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_simbench import steps_of, true_chain, sim_prompt, parse_chain  # noqa: F401
