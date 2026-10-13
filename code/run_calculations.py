from pathlib import Path
import subprocess, sys
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
jobs = [('4_1_理论算例', 'analyze_theory.py'), ('4_1_理论算例', 'analyze_method_comparison.py'), ('4_2_BOPTEST', 'source/prepare_data.py'), ('4_2_BOPTEST', 'source/compute_max_deviation.py')]
for folder, script in jobs:
    print(f'Running {folder}/{script}', flush=True)
    subprocess.run([sys.executable, '-B', '-X', 'utf8', script], cwd=root/folder, check=True)
print('All calculations completed.')
