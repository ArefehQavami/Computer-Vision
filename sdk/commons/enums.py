from enum import Enum

class Models(Enum):
    FNET512 = "Facenet512"


class Metrics(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    L2 = "euclidean_l2"


class Detector(Enum):
    OCV = "opencv"
    MTCNN = "mtcnn"
    RFACE = "retinaface"


