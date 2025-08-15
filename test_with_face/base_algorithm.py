import os
from dotenv import load_dotenv
from django.conf import settings
from keras.models import load_model
from server.settings import BASE_DIR
import tensorflow as tf

load_dotenv(BASE_DIR / "server/.env")


class Algorithm:
    def __init__(self):
        pass

    def load_model(self, model_path=''):
        if model_path:
            return load_model(self.model_path)

    def get_embedding_vector(self, image):
        pass


class Singleton:
    @classmethod
    def __new__(cls, *args, **kwargs):
        try:
            if f'django-insecure-{os.getenv("SECRET_KEY")}' != settings.SECRET_KEY:
                raise Exception('SECRET KEY NOT FOUND!!!!!')
            if not hasattr(cls, 'instance'):
                cls.instance = super(Singleton, cls).__new__(cls)
            return cls.instance
        except Exception as e:
            print(e)


def check_pass(func):
    def wrapper():
        try:
            if f'django-insecure-{os.getenv("SECRET_KEY")}' != settings.SECRET_KEY:
                raise Exception('SECRET KEY NOT FOUND!!!!!')
            func()
        except Exception as e:
            print(e)

    return wrapper


def check_pass_with_args(func):
    def wrapper_func(*args, **kwargs):
        try:
            if f'django-insecure-{os.getenv("SECRET_KEY")}' != settings.SECRET_KEY:
                print('SECRET!!!!')
                raise Exception('SECRET KEY NOT FOUND!!!!!')
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return wrapper_func
