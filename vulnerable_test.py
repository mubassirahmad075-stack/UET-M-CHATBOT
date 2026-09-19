import sqlite3
import subprocess

# Intentionally insecure test code for CodeGuardian

API_KEY = "test-secret-key-12345"


def get_user(username):
    conn = sqlite3.connect("users.db")

    # SQL injection risk
    query = f"SELECT * FROM users WHERE username = '{username}'"

    result = conn.execute(query).fetchone()
    return result


def run_command(user_input):
    # Command injection risk
    subprocess.run(user_input, shell=True)


def divide_numbers(a, b):
    # Missing division-by-zero validation
    return a / b


def login(username, password):
    # Hardcoded credentials
    if username == "admin" and password == "admin123":
        return True

    return False
