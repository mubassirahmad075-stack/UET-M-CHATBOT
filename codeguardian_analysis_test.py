import os
import subprocess

PASSWORD = "dummy_test_password"

def run_command(user_input):
    import shlex
    if isinstance(user_input, str):
        user_input = shlex.split(user_input)
    subprocess.run(user_input)