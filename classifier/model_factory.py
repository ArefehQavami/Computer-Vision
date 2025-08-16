import timm
import torch.nn as nn
from config import Config


class ModelFactory:
    @staticmethod
    def create_model():
        model = timm.create_model(Config.MODEL_NAME, pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        for block in model.stages[-6:]:
            for param in block.parameters():
                param.requires_grad = True

        model.reset_classifier(num_classes=Config.NUM_CLASSES)
        for param in model.head.parameters():
            param.requires_grad = True

        return model.to(Config.DEVICE)

