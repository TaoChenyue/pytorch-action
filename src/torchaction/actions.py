from typing import *
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from tensorboardX import SummaryWriter
from .parse_args import parse_args
from .meanloss import MeanLoss


def compare_metrics(p1, p2, names):
    for name in names:
        if p1[name] > p2[name]:
            return True
    return False


class BasicAction:
    def __init__(self) -> None:
        self.parse(parse_args())
        self.init_paras()

    def check_device(self, gpus):
        gpus=[int(x) for x in gpus.split(',')] if gpus else []
        gpu_nums=len(gpus)
        use_device='cpu'
        if gpu_nums:
            assert torch.cuda.is_available(),\
                "Please update torch's version with cuda, you can download torch-cuda versions in https://pytorch.org/. "
            if gpu_nums>1:
                self.GPUS=gpus
            use_device="cuda:{}".format(gpus[0])
        print("Using Device {},model load into device {}".format(gpus,use_device))
        return torch.device(use_device)

    def parse(self, args):
        # dataset
        self.ROOT:str = args.root
        self.NUM_CLASSES:int = args.num_classes
        self.SPLIT_RATE:float = args.split_rate
        # train
        self.CKPT:str = args.ckpt
        self.GPUS:list = []
        self.DEVICE = self.check_device(args.gpus)
        self.EPOCHS:int = args.epochs
        self.BATCH_SIZE:int = args.batch_size
        self.LR:float = args.lr
        # save
        self.SAVE_NAME:str = args.save_name
        self.SAVE_SUFFIX:str = args.save_suffix

    def init_paras(self):
        # parameters
        self.START_EPOCH:int = 0
        # variables
        self.dataset: Dataset = None
        self.model: torch.nn.Module = None
        self.loss_fun: Any = None
        self.optimizer: Any = None
        self.output_formater = lambda x: x
        self.train_metrics: Dict[str, Any] = None
        self.valid_metrics: Dict[str, Any] = None
        self.best_ckpt_selectors: List = None

    def prepare_data(self):
        dataset_length = len(self.dataset)
        train_dataset_length = round(self.SPLIT_RATE * dataset_length)
        valid_dataset_length = dataset_length - train_dataset_length
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_dataset_length, valid_dataset_length]
        )
        
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset, batch_size=self.BATCH_SIZE, shuffle=False
        )

    def prepare_model(self):
        if self.CKPT:
            checkpoint = torch.load(self.CKPT,map_location=torch.device(self.DEVICE))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.START_EPOCH = checkpoint["epoch"]

        if self.GPUS:
            self.model=torch.nn.DataParallel(self.model,device_ids=self.GPUS)
        self.model.to(self.DEVICE)

    def prepare_metrics(self):
        self.train_loss = MeanLoss(self.loss_fun)
        self.valid_loss = MeanLoss(self.loss_fun)
        self.best_metrics = {}
        self.last_metrics = {}
        for name in self.train_metrics.keys():
            self.best_metrics.setdefault(name, 0)
            self.last_metrics.setdefault(name, 0)
        self.train_writer = SummaryWriter("logs/{}/train".format(self.SAVE_NAME))
        self.valid_writer = SummaryWriter("logs/{}/valid".format(self.SAVE_NAME))

    def train(self):
        assert self.dataset, "Dataset not found"
        assert self.model, "Model not set"
        assert self.loss_fun, "loss function should be defined"
        assert self.optimizer, "optimizer should be defined"
        self.prepare_data()
        self.prepare_model()
        self.prepare_metrics()
        for epoch in range(self.EPOCHS):
            if epoch < self.START_EPOCH:
                continue
            self.run_single_epoch(epoch, train=True)
            self.run_single_epoch(epoch, train=False)
            self.record_epoch(epoch)

    def run_single_epoch(self, epoch, train=True):
        self.model.train() if train else self.model.eval()
        bar = tqdm(self.train_dataloader if train else self.val_dataloader)
        for batch, (x, y) in enumerate(bar):
            self.optimizer.zero_grad()
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE)
            output = self.model(x)
            loss_mode = self.train_loss if train else self.valid_loss
            loss_val = loss_mode(output, y)
            self.optimizer.step()
            # batch metric
            self.train_loss
            y_predict = self.output_formater(output)
            bar.desc = "mode:{},epoch:{}/{},batch:{}/{},loss:{:.3f}".format(
                "train" if train else "valid",
                epoch + 1,
                self.EPOCHS,
                batch + 1,
                len(bar),
                loss_val,
            )
            metrics = self.train_metrics if train else self.valid_metrics
            for metric in metrics.values():
                metric.update(y_predict.to('cpu'), y.to('cpu'))

    def record_epoch(self, epoch):
        self.train_writer.add_scalar("loss", self.train_loss.compute(), epoch)
        self.valid_writer.add_scalar("loss", self.valid_loss.compute(), epoch)
        self.train_loss.reset()
        self.valid_loss.reset()
        for name, metric in self.train_metrics.items():
            val = metric.compute()
            metric.reset()
            self.last_metrics[name] = val
            self.train_writer.add_scalar(name, val, epoch)

        for name, metric in self.valid_metrics.items():
            self.valid_writer.add_scalar(name, metric.compute(), epoch)
            metric.reset()
        self.save_best_ckpt(epoch)
        self.save_ckpt(epoch, "last")

    def save_ckpt(self, epoch, mode):
        Path("weights").mkdir(parents=True, exist_ok=True)
        if self.GPUS:
            out_model=self.model.module
        else:
            out_model=self.model
        torch.save(
            {
                "model_state_dict": out_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            "weights/{}_ckpt_{}.{}".format(mode, self.SAVE_NAME, self.SAVE_SUFFIX),
        )

    def save_best_ckpt(self, epoch):
        if compare_metrics(
            self.last_metrics, self.best_metrics, self.best_ckpt_selectors
        ):
            self.best_metrics = self.last_metrics.copy()
            self.save_ckpt(epoch, "best")
