import torch.nn as nn
import torch.optim as optim
import torch
from config import Config
from dataloader_factory import DataLoaderFactory
from model_factory import ModelFactory
from trainer import Trainer
from evaluator import Evaluator
from onnx_exporter import ONNXExporter


def main():
    # Load data
    print("Preparing dataloaders...")
    dataloaders = DataLoaderFactory().get_loaders()

    # Create model
    print("Creating model...")
    model = ModelFactory.create_model()

    # # Define optimizer, criterion, scheduler
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # # Train model
    # print("Starting training...")
    # trainer = Trainer(model, dataloaders, criterion, optimizer, scheduler)
    # trained_model = trainer.train()

    # Load best model and evaluate
    print("Evaluating best model...")
    model.load_state_dict(torch.load(f"{Config.MODEL_SAVE_NAME}.pth"))
    standard_cm = Evaluator.evaluate(model, dataloaders["standard_test"], "STANDARD TEST SET")
    # device_cm = Evaluator.evaluate(model, dataloaders["device_test"], "DEVICE TEST SET (REAL-WORLD)")
    Evaluator.save_confusion_matrix_image(standard_cm, Config.CLASS_NAMES, "Standard_Test_Confusion_Matrix.png")
    # Evaluator.save_confusion_matrix_image(device_cm, Config.CLASS_NAMES, "Device_Test_Confusion_Matrix.png")


    # Export to ONNX
    print("Exporting to ONNX...")
    ONNXExporter.export(model)


if __name__ == "__main__":
    main()

