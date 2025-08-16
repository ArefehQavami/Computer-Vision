import torch
from sklearn.metrics import classification_report, confusion_matrix
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import os

class Evaluator:
    @staticmethod
    def evaluate(model, dataloader, test_name="Test"):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(Config.DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print(f"\n{'=' * 40}\n{test_name} Results\n{'=' * 40}")
        print("Classification Report:")
        print(Config.CLASS_NAMES)
        print(classification_report(all_labels, all_preds, target_names=Config.CLASS_NAMES))

        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        Evaluator.print_confusion_matrix(cm)
        return cm

    @staticmethod
    def print_confusion_matrix(cm):
        header = f"{'':<15}" + ''.join([f"{name:<15}" for name in Config.CLASS_NAMES])
        print(header)
        for i, row in enumerate(cm):
            row_str = f"{Config.CLASS_NAMES[i]:<15}" + ''.join([f"{val:<15}" for val in row])
            print(row_str)

    @staticmethod
    def save_confusion_matrix_image(cm, class_names, filename):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set(xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title=filename[:-4])

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(os.path.join("results", filename))
        plt.close()


