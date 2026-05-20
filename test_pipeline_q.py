import os
import sys
import subprocess

# Set working directory to the experiments root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE

def run_small_test():
    # Modify main.py temporarily to run a tiny test
    main_path = os.path.join(_ROOT, 'new_experiments', 'var_study_q', 'main.py')
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Tiny test: 1 dataset, 2 estimators, 2 block sizes
    original_data = "DATASET_KEYS = ['c', 'f', 'ca', 's']"
    temp_data = "DATASET_KEYS = ['c']"
    original_est = "N_ESTIMATORS = 10"
    temp_est = "N_ESTIMATORS = 2"
    original_blocks = "BLOCK_SIZES  = [1, 5, 10, 15, 20, 30, 35, 40]"
    temp_blocks = "BLOCK_SIZES  = [5, 20]"
    
    modified_content = content.replace(original_data, temp_data)
    modified_content = modified_content.replace(original_est, temp_est)
    modified_content = modified_content.replace(original_blocks, temp_blocks)
    
    temp_main = os.path.join(_ROOT, 'new_experiments', 'var_study_q', 'main_test.py')
    with open(temp_main, 'w') as f:
        f.write(modified_content)
    
    print("Starting small pipeline test...")
    try:
        # Use venv python if it exists
        venv_python = os.path.join(_ROOT, '.venv', 'Scripts', 'python.exe')
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        print(f"Using python: {python_exe}")
        
        res = subprocess.run([python_exe, temp_main], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print("ERROR in pipeline test:")
            print(res.stderr)
        else:
            print("Pipeline test successful!")
    finally:
        if os.path.exists(temp_main):
            os.remove(temp_main)

if __name__ == "__main__":
    run_small_test()
