import subprocess

def run_A(input):
    command = f"time python3 hw1-q2.py mlp -learning_rate {input} -batch_size 16"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

vartiable_inputs = [0.001, 0.01, 0.1, 1]

with open('2-2-b.txt', 'w') as file:
    for input in vartiable_inputs:
        stdout, stderr = run_A(input)
        file.write(f"Learning Rate: {input}\n")
        file.write(f"=== STDOUT ===\n{stdout}\n")
        file.write(f"=== STDERR ===\n{stderr}\n")
        file.write("=" * 40 + "\n")
print("Done!")