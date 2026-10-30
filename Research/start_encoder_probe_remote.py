"""Small bootstrap on the compute node; no head-node execution."""
from pathlib import Path
import socket
import subprocess
import sys

root=Path('/tmp/shihangz/photo-tone-transfer')
assert socket.gethostname().split('.')[0]=='trinity-2-23'
source=root/'Research/dinov2_pkg';target=root/'Research/External_Code/dinov2/dinov2'
if source.exists() and not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
base=root/'Experiments/encoder_probe_20261030';base.mkdir(parents=True,exist_ok=True)
phase=sys.argv[1]
command=f'cd {root} && {sys.executable} -u Model/encoder_experiment_jobs.py --root {root} --phase {phase} > {base}/{phase}_orchestration.log 2>&1'
subprocess.run(['tmux','new-session','-d','-s','encoder-probe-'+phase,command],check=True)
print('started','encoder-probe-'+phase,base,flush=True)
