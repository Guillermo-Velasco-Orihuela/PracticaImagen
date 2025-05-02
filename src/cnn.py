import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tempfile import TemporaryDirectory

class CNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Replace the final classification layer
        self.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        # Remove the original classifier head
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_model(self, 
                    train_loader, 
                    valid_loader, 
                    optimizer, 
                    criterion, 
                    epochs, 
                    nepochs_to_save=10,
                    device=torch.device("cuda")):

        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()

        with TemporaryDirectory() as temp_dir:
            best_model_path = os.path.join(temp_dir, 'best_model.pt')
            best_accuracy = 0.0
            self.to(device)
            torch.save(self.state_dict(), best_model_path)

            history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
            for epoch in range(epochs):
                self.train()
                train_loss = 0.0
                train_accuracy = 0.0
                for images, labels in train_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with autocast():
                        outputs = self(images)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()
                    train_accuracy += (outputs.argmax(1) == labels).sum().item()

                train_loss /= len(train_loader)
                train_accuracy /= len(train_loader.dataset)
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)

                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

                self.eval()
                valid_loss = 0.0
                valid_accuracy = 0.0
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        with autocast():
                            outputs = self(images)
                            loss = criterion(outputs, labels)
                        valid_loss += loss.item()
                        valid_accuracy += (outputs.argmax(1) == labels).sum().item()

                valid_loss /= len(valid_loader)
                valid_accuracy /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)
                history['valid_accuracy'].append(valid_accuracy)

                print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {valid_loss:.4f}, Val Acc: {valid_accuracy:.4f}")

                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    torch.save(self.state_dict(), best_model_path)

            self.load_state_dict(torch.load(best_model_path))
            return history

    def predict(self, data_loader, device=torch.device("cuda")):
        self.eval()
        self.to(device)
        predictions = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                outputs = self(images)
                predictions.extend(outputs.argmax(1).tolist())
        return predictions

    def save_model(self, filename: str):
        os.makedirs(os.path.dirname('models/'), exist_ok=True)
        filename = os.path.join('models', filename)
        torch.save(self.state_dict(), filename + '.pt')

    @staticmethod
    def _plot_training(history):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['valid_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


def load_data(train_dir, valid_dir, batch_size, img_size):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, valid_loader, len(train_dataset.classes)
