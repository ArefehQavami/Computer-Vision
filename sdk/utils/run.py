import os


def run():
    os.system('gunicorn main:app')
