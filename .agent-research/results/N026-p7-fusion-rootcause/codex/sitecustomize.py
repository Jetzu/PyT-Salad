"""Load the N026 test patch only in requested entry points."""
import os,sys
if os.environ.get('N026_PATCH')=='barrier' and sys.argv[0].endswith(('generate.py','bench_p7.py','barrier_forward.py','test_model_parity.py')):
    import patch_runtime
    patch_runtime.install()
