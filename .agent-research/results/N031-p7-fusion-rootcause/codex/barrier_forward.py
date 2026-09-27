"""Run N025-derived forward reproducer with the local graph-break patch."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import patch_runtime
patch_runtime.install()
sys.argv[0]=str(Path(__file__).parent/'bisect_forward.py')
runpy.run_path(sys.argv[0],run_name='__main__')
