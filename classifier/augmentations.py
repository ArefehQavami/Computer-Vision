from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp, vp = (max_wh - w) // 2, (max_wh - h) // 2
        padding = [hp, vp, hp, vp]
        return TF.pad(image, padding, 0, 'constant')


class AugmentationFactory:
    @staticmethod
    def get_transforms(strength, input_size):
        base = [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
        ]
        if strength == "medium":
            base += [
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.RandomAffine(15, translate=(0.1, 0.1)),
            ]
        elif strength == "high":
            base += [
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.RandomAffine(30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomPerspective(0.3),
                transforms.RandomRotation((0, 180)),
                transforms.RandomVerticalFlip(0.2),
            ]
        return transforms.Compose(base + [transforms.ToTensor()])

