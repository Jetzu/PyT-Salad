import importlib.util
for name in ('model','sampling','stack','encoder'):
 path='tests/test_'+name+'_parity.py'
 s=importlib.util.spec_from_file_location('repro_'+name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
 old=m.os.path.exists; m.os.path.exists=lambda p: False
 try: print(name,'missing return',m.main(),flush=True)
 finally: m.os.path.exists=old
