from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from lambda_mobilenetv1 import get_mobilenetv1
from utils import (
    AverageMeter,
    accuracy,
    create_exp_dir,
    count_parameters_in_MB,
    save_checkpoint,
)


parser = argparse.ArgumentParser(description="Train Lambda MobileNet-V1")

parser.add_argument(
    "--save_root", type=str, default="./", help="models and logs are saved here"
)
parser.add_argument(
    "--img_root", type=str, default="./datasets", help="path name of image dataset"
)

parser.add_argument("--num_class", type=int, default=10, help="number of classes")
parser.add_argument(
    "--epochs", type=int, default=120, help="number of total epochs to run"
)
parser.add_argument("--batch_size", type=int, default=128, help="The size of batch")
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--seed", type=int, default=666, help="random seed")

parser.add_argument(
    "--data_name", type=str, required=True, help="name of dataset"
)  # cifar10/cifar100

parser.add_argument(
    "--net_name",
    type=str,
    required=True,
    help="Name of network (only mobilenetv1 for now)",
)

parser.add_argument(
    "--saved_model_name",
    type=str,
    default="saved_model",
    help="name of the model to be saved",
)

args, unparsed = parser.parse_known_args()

create_exp_dir(args.img_root)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    print("Setting the seed for generating random numbers for the current GPU")
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def _init_fn(worker_id):
    # set in Dataloader parameter worker_init_fn=_init_fn
    np.random.seed(int(args.seed))


def main():
    if args.data_name == "cifar100":
        classes = 100
    else:
        classes = 10

    net = get_mobilenetv1(num_classes=classes)
    print(net)
    print("MobileNetV1: ", count_parameters_in_MB(net))

    if args.cuda:
        net = nn.DataParallel(net).cuda()

    # initialize optimizer
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    def lmbda(epoch):
        return 0.98

    if "mobilenetv1" in args.net_name:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90], gamma=0.1
        )
    else:
        raise Exception("Invalid model name...")

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # define transforms
    if args.data_name == "cifar100":
        dataset = dst.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif args.data_name == "cifar10":
        dataset = dst.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        raise Exception("Invalid dataset name...")

    if args.data_name in ["cifar100", "cifar10"]:
        train_transform = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        # define data loader
        train_loader = torch.utils.data.DataLoader(
            dataset(
                root=args.img_root, transform=train_transform, train=True, download=True
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            worker_init_fn=_init_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset(
                root=args.img_root, transform=test_transform, train=False, download=True
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            worker_init_fn=_init_fn,
        )
    else:
        raise ValueError(f"Invalid dataset")

    best_top1 = 0

    print("starting training...", flush=True)

    for epoch in range(1, args.epochs + 1):

        epoch_start_time = time.time()
        tr_loss, tr_top1, tr_top5 = train(
            train_loader, net, optimizer, criterion, epoch
        )

        # evaluate on testing set
        test_loss, test_top1, test_top5 = test(test_loader, net, criterion)

        scheduler.step()

        epoch_duration = time.time() - epoch_start_time

        temp_lr = None
        for param_group in optimizer.param_groups:
            temp_lr = param_group["lr"]

        print(
            f"Ep: {epoch} eptime: {epoch_duration:.2f} TrLoss: {tr_loss:.4f}  TrTop1: {tr_top1:.2f} TrTop5: {tr_top5:.2f}  EvLoss: {test_loss:.2f}  EvTop1: {test_top1:.2f}  EvTop5: {test_top5:.2f} lr: {temp_lr:.5f}",
            flush=True,
        )

        # save model
        is_best = False
        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top5 = test_top5
            is_best = True

        save_checkpoint(
            {
                "epoch": epoch,
                "net": net.state_dict(),
                "prec@1": test_top1,
                "prec@5": test_top5,
            },
            0,
            is_best,
            args.save_root,
            args.saved_model_name,
        )


def train(train_loader, net, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        out = net(img)
        loss = criterion(out, target)
        prec1, prec5 = accuracy(out, target, topk=(1, 5))

        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg, top5.avg


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            out = net(img)
            loss = criterion(out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
