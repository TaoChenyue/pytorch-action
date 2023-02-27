import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.models import resnet34
from torchvision import transforms
import torchmetrics
from torchaction.actions import BasicAction
import random

project = BasicAction()
# prepare data
transform = transforms.ToTensor()
target_transform = lambda x: torch.tensor(x, dtype=torch.long)
project.dataset = MNIST(
    root="datasets",
    train=True,
    download=True,
    transform=transform,
    target_transform=target_transform,
)
indices=list(range(len(project.dataset)))
random.shuffle(indices)
project.dataset = Subset(project.dataset,indices[:100])
# prepare model
project.model = resnet34()
project.model.fc = torch.nn.Linear(
    project.model.fc.in_features, project.NUM_CLASSES
)
project.model = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3, 1), project.model)
# prepare train
project.loss_fun=torch.nn.CrossEntropyLoss()
project.optimizer_fun=torch.optim.Adam
# prepare metrics
project.output_formater=lambda x: torch.max(x,dim=1).indices
project.train_metrics={
    'acc':torchmetrics.Accuracy(task="multiclass",num_classes=project.NUM_CLASSES),
    'prec':torchmetrics.Precision(task="multiclass",num_classes=project.NUM_CLASSES),
    'recall':torchmetrics.Recall(task="multiclass",num_classes=project.NUM_CLASSES),
    'f1':torchmetrics.F1Score(task="multiclass",num_classes=project.NUM_CLASSES),
}
project.valid_metrics={
    'acc':torchmetrics.Accuracy(task="multiclass",num_classes=project.NUM_CLASSES),
    'prec':torchmetrics.Precision(task="multiclass",num_classes=project.NUM_CLASSES),
    'recall':torchmetrics.Recall(task="multiclass",num_classes=project.NUM_CLASSES),
    'f1':torchmetrics.F1Score(task="multiclass",num_classes=project.NUM_CLASSES),
}
# prepare save
project.best_ckpt_selectors=['acc']
project.train()
