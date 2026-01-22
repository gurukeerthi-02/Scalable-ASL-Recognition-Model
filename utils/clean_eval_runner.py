import os
import subprocess

def run_and_save(script, output_file):
    with open(output_file, 'w', encoding='ascii', errors='ignore') as f:
        process = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            # Filter out non-ascii characters manually
            clean_line = ''.join(i for i in line if ord(i) < 128)
            f.write(clean_line)
            print(clean_line, end='')
        process.wait()

print("Running evaluate-model.py...")
run_and_save('evaluate-model.py', 'eval_static_clean.txt')
print("\nRunning evaluate-model-lstm.py...")
run_and_save('evaluate-model-lstm.py', 'eval_dynamic_clean.txt')
