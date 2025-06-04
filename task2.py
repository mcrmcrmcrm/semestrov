import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F


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

        # ResNet слои
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


model = ResNet34()
model.load_state_dict(torch.load('resnet34.pth'))
model.eval()


def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_t = transform(img).unsqueeze(0)  # Добавляем размерность batch

    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    class_names = ['cat', 'dog']  # Должно совпадать с порядком при обучении
    return class_names[pred.item()], confidence.item()


image_path = 'test_image.jpg'  # Путь к вашему изображению
prediction, accuracy = predict(image_path)
print(f'Это {prediction}! Точность: {accuracy*100:.2f}%')