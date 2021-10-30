from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import yaml
import argparse

opt = argparse.Namespace(**yaml.load(open("config.yml"), yaml.FullLoader))

# 使用torchvision预定义好的Dataset, 自定义Dataset在此处暂时不涉及
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

# import matplotlib.pyplot as plt
# # image: (3, 32, 32) float [0~1] tensor, label: int
# image, label = training_data[1]
# figure = plt.figure(figsize=(2, 2))
# plt.imshow(image.numpy().transpose([1, 2, 0]))


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# for x, y in train_dataloader:
#     pass  # x is (64, 3, 32, 32) float cpu tensor, y is (64,) cpu LongTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 32, 16, 16)

            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),  # (B, 64, 8, 8)
            
            nn.Conv2d(64, 16, 1, stride=1, padding=0, bias=True),
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(16*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

model = NeuralNetwork()
loss_fn = 

checkpoint = torch.load(args.save_dir)

if args.from_pretrain:
    model.load_state_dict()

if args.freeze > 0:
    # 冻结参数
    pass

current_epoch = checkpoint if args.from_pretrain else 0
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

model.train()

for epoch in range(current_epoch, min(args.train_epoches, args.total_epoches)):
    for x, target in train_dataloader:
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            output = net(x)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
torch.save()

# torchscript save

# TODO: 多GPU训练