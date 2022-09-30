import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        h1 = nn.functional.relu(self.layer1(x))
        h2 = nn.functional.relu(self.layer2(h1))
        h3 = self.dropout(h1 + h2)
        logits = self.layer3(h3)
        return logits

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    # def backward(self, trainer, loss, optimizer, optimizer_idx):
    #     loss.backward()


    # 只下载一次
    # def prepare_data(self) -> None:
    #     datasets.MNIST(
    #         'mnist',
    #         train=True,
    #         download=True,
    #         transform=transforms.ToTensor()
    #     )

    #     datasets.MNIST(
    #         'mnist',
    #         train=False,
    #         download=True,
    #         transform=transforms.ToTensor()
    #     )
    

    # def setup(self, stage: Optional[str] = None) -> None:
    #     self.train_dataset = datasets.MNIST(
    #         'mnist',
    #         train=True,
    #         download=True,
    #         transform=transforms.ToTensor()
    #     )

    #     self.val_dataset = datasets.MNIST(
    #         'mnist',
    #         train=False,
    #         download=True,
    #         transform=transforms.ToTensor()
    #     )

    def training_step(self, batch, batch_idx):
        x, y = batch
        b = x.size(0)
        x = x.view(b, -1)
        logits = self(x)
        loss = self.loss(logits, y)
        # acc = pl.metrics.functional.accuracy(logits, y)
        acc = loss / 2
        return {
            "loss": loss,
            "progress_bar": {
                "train_acc": acc
            },  # no effect now
            # "log": loss  # ?
        }
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results["progress_bar"]["val_acc"] = results["progress_bar"]["train_acc"]
        del results["progress_bar"]["train_acc"]
        return results

    # val_step_outputs is deprecated?
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x["loss"] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x["progress_bar"]["val_acc"] for x in val_step_outputs]).mean()
        return {
            "val_loss": avg_val_loss,
            "progress_bar": {
                "avg_val_acc": avg_val_acc
            }
        }

    def train_dataloader(self):
        train_dataset = datasets.MNIST(
            'mnist',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        train_loader = DataLoader(train_dataset, batch_size=32)
        return train_loader

    def val_dataloader(self):
        val_dataset = datasets.MNIST(
            'mnist',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        val_loader = DataLoader(val_dataset, batch_size=32)
        return val_loader

model = Classifier()
trainer = pl.Trainer(
    max_epochs=10,
    # gpus=1,
)
trainer.fit(model)