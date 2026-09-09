import os
import subprocess

PASSWORD = "dummy_test_password"

def run_command(user_input):
    subprocess.run(user_input, shell=True)