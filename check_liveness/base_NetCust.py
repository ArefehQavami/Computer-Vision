import os
import base_Net1
import commons_functions


def loadModel():
    try:

        model = base_Net1.DNN(dimension=512)
        model.load_weights('E:/Face Verification/senet-face/code/check_liveness/net_weights.h5')
        return model
    except Exception as e:
        raise e
