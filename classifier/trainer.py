import torch
from early_stopping import EarlyStopping
from config import Config


class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler=None):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopper = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            delta=Config.EARLY_STOPPING_DELTA,
            verbose=True
        )

    def train(self):
        best_acc = 0.0

        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
            for phase in ["train", "val"]:
                self.model.train() if phase == "train" else self.model.eval()

                running_loss, running_corrects = 0.0, 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model.state_dict(), f"{Config.MODEL_SAVE_NAME}.pth")
                        print("Best model saved.")

            if self.scheduler:
                self.scheduler.step()

            self.early_stopper(epoch_acc)
            if self.early_stopper.early_stop:
                print("Early stopping activated.")
                break

        return self.model

