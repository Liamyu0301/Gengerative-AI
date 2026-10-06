"""HW1 Q4: train variants one after another on the same GPU, logging to logs/<run>.log.

Usage:  python run_all.py --device=cuda                      # all 12 runs
        python run_all.py --device=mps baseline rope         # pick runs; extra --k=v go to train.py
Every run uses seed 1337 and max_iters 5000 (set in the configs); only the named change differs.
"""
import subprocess, sys, os

VARIANTS = ['baseline', 'rmsnorm', 'swiglu', 'nope', 'rope', 'gqa']
RUNS = VARIANTS + [v + '_lr2e-3' for v in VARIANTS]

overrides = [a for a in sys.argv[1:] if a.startswith('--')]
runs = [a for a in sys.argv[1:] if not a.startswith('--')] or RUNS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('logs', exist_ok=True)
for run in runs:
    cmd = [sys.executable, '-u', 'train.py', f'config/hw1_{run}.py', *overrides]
    print('===', ' '.join(cmd), flush=True)
    with open(f'logs/{run}.log', 'w') as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line); log.write(line); log.flush()
        if proc.wait() != 0:
            sys.exit(f'{run} failed, see logs/{run}.log')
