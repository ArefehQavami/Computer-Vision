import os

def start():
    os.system('celery -A server beat -l info -f ./logs/master.log --detach')
    os.system('celery -A server worker -l info -f ./logs/agents.log --detach')
