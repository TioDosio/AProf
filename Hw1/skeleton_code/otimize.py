import subprocess

def run_program(learning_rate):
    command = f"time python hw1-q2.py mlp -learning_rate {learning_rate}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

learning_rates = [1, 0.1, 0.01]

with open('output.txt', 'w') as file:
    for rate in learning_rates:
        stdout, stderr = run_program(rate)
        file.write(f"Learning Rate: {rate}\n")
        file.write(f"=== STDOUT ===\n{stdout}\n")
        file.write(f"=== STDERR ===\n{stderr}\n")
        file.write("=" * 40 + "\n")
