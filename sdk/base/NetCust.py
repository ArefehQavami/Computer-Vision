import os
from base import Net
from commons import functions
from base.algorithm import BASE_DIR
from base.algorithm import settings
from base.algorithm import load_dotenv

load_dotenv(BASE_DIR / "server/.env")

def loadModel():
    try:
        if f'django-insecure-{os.getenv("SECRET_KEY")}' != settings.SECRET_KEY:
            raise Exception('SECRET KEY NOT FOUND!!!!!')

        model = Net.DNN(dimension=512)
        home = functions.get_home()
        if not os.path.isfile(f'{home}/.deep/weights/net_weights.h5'):
            print("NOT FOUND!!!!!")
        model.load_weights(home + '/.deep/weights/net_weights.h5')
        return model
    except Exception as e:
        raise e
