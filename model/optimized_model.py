import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_transforms() -> Dict[str, transforms.Compose]:
    return {
        split: transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) for split in ['train', 'valid', 'test']
    }

def load_data(data_dir: str, batch_size: int, transforms: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datasets_dict = {
        split: datasets.ImageFolder(root=f"{data_dir}/{split}", transform=transforms[split])
        for split in ['train', 'valid', 'test']
    }
    
    return (
        DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True, num_workers=50),
        DataLoader(datasets_dict['valid'], batch_size=batch_size, shuffle=False, num_workers=50),
        DataLoader(datasets_dict['test'], batch_size=batch_size, shuffle=False, num_workers=50)
    )

class ResNetWithClassifier(nn.Module):
    def __init__(self, num_classes: int = 196):
        super().__init__()
        self.backbone = models.resnet152(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

def plot_training_results(train_losses: list, valid_losses: list, valid_accuracies: list, num_epochs: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, num_epochs + 1)
    # Loss plot
    ax1.plot(epochs, train_losses, label='Training Loss')
    ax1.plot(epochs, valid_losses, label='Validation Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, valid_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs: int, device: torch.device):
    train_losses, valid_losses, valid_accuracies = [], [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = total = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(epoch_valid_loss)
        accuracy = correct / total
        valid_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_valid_loss:.4f}, Accuracy: {accuracy:.4f}")

    plot_training_results(train_losses, valid_losses, valid_accuracies, num_epochs)
    return model

def main():
    # Configuration
    DATA_DIR = "Kevin-MIE1517-processeddata"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    NUM_CLASSES = 196
    
    # Setup
    device = get_device()
    transforms_dict = create_transforms()
    train_loader, valid_loader, test_loader = load_data(DATA_DIR, BATCH_SIZE, transforms_dict)
    
    # Model initialization
    model = ResNetWithClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Training
    model = train_model(model, criterion, optimizer, train_loader, valid_loader, NUM_EPOCHS, device)
    
    # Save the model
    torch.save(model.state_dict(), 'resnet152_classifier.pth')

if __name__ == "__main__":
    main()