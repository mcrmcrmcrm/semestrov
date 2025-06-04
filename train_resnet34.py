import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
import time


# 1. Определение архитектуры ResNet34 (с ручным указанием слоёв)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super().__init__()
        self.in_channels = 64

        # Первые слои
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet слои (вручную указываем количество)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 3 слоя
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 4 слоя
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 6 слоёв
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 3 слоя

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet34(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# 2. Подготовка данных
def prepare_data(data_dir='data'):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes


# 3. Функции для обучения и валидации
def train_model(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}]'
                  f'\tLoss: {loss.item():.4f}')

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_f1 = f1_score(all_labels, all_preds, average='macro')

    return train_loss, train_acc, train_f1


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return val_loss, val_acc, val_f1


# 4. Основной цикл обучения
def main():
    # Настройки
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    learning_rate = 0.001

    # Подготовка данных
    train_loader, val_loader, classes = prepare_data()
    print(f"Classes: {classes}")

    # Инициализация модели
    model = ResNet34(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    print("Starting training...")
    best_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Обучение
        train_loss, train_acc, train_f1 = train_model(model, train_loader, criterion, optimizer, epoch, device)

        # Валидация
        val_loss, val_acc, val_f1 = validate_model(model, val_loader, criterion, device)

        # Логирование
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}')
        print('-' * 60)

        # Сохранение лучшей модели
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'resnet34.pth')
            print(f'New best model saved with F1: {best_f1:.4f}')

        # Регулировка learning rate
        scheduler.step(val_f1)

    print(f'Training complete. Best F1: {best_f1:.4f}')

if __name__ == '__main__':
    main()