import os
import base_Net
import commons_functions


def loadModel():
    try:

        model = base_Net.DNN(dimension=512)
        home = commons_functions.get_home()
        model.load_weights('E:/Face Verification/senet-face/code/test_with_face/net_weights.h5')
        return model
    except Exception as e:
        raise e
