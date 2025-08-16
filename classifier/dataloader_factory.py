import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config
from augmentations import AugmentationFactory


class DataLoaderFactory:
    def __init__(self):
        self.train_transforms = AugmentationFactory.get_transforms(Config.AUGMENTATION_STRENGTH, Config.INPUT_SIZE)
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
            transforms.ToTensor()
        ])

    def get_loaders(self):
        data_dir = Config.DATA_DIR

        datasets_map = {
            "train": datasets.ImageFolder(os.path.join(data_dir, "train"), transform=self.train_transforms),
            "val": datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=self.val_test_transforms),
            "standard_test": datasets.ImageFolder(os.path.join(data_dir, "test"), transform=self.val_test_transforms),
            # "device_test": datasets.ImageFolder(os.path.join(data_dir, "device_test"), transform=self.val_test_transforms),
        }

        dataloaders = {
            phase: DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=(phase == "train"))
            for phase, dataset in datasets_map.items()
        }

        return dataloaders

