#!/bin/bash


celery -A server beat -l info -f logger/master.log --detach
celery -A server worker -l info -f logger/agents.log -n 1 --detach
gunicorn main:app