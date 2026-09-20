import subprocess
import sqlite3

PASSWORD = "demo-password-123"


def get_user(username):
    connection = sqlite3.connect("users.db")
    cursor = connection.cursor()

    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)

    return cursor.fetchall()


def run_command(command):
    subprocess.run(command, shell=True)


def calculate(expression):
    return eval(expression)


def divide(a, b):
    return a / b
