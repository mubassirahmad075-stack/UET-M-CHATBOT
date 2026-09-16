import subprocess

PASSWORD = "demo_password_123"

def run_command(user_input):
    subprocess.run(user_input, shell=True)
