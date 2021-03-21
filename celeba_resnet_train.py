import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.optim.lr_scheduler import (ReduceLROnPlateau, LambdaLR,
                                      MultiplicativeLR)
from celeba import CelebA
import time
import json
import os
import argparse
from pathlib import Path

CELEBA_DIR = "./CelebA"
UNALIGNED_DIR = "./img_celeba"
SPLIT_DIR = "new-splits"
LOG_DIR = "logs"
MODEL_DIR = "resnet-model"

class AttributeNN(nn.Module):
    """Base network for learning representations. Just a wrapper for
    resnnet 18 which maps the last layer to 40 outputs instead of the
    1000 used for ImageNet classification.
    """
    def __init__(self, n_labels, pretrain=False):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrain)
        self.fc_in_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(self.fc_in_feats, n_labels, bias=True)
        self.n_labels = n_labels

    def forward(self, input):
        output = self.resnet(input)
        return output

class TrainLog():
    """Logger class for logging training metrics to a file. Set
    verbose=False to disable printing to terminal.
    """
    def __init__(self, journal_file="train_log.txt", verbose=True):
        self.journal = [{"train" : []}]
        self.journal_file = journal_file
        self.verbose = verbose

    def start_log(self):
        self.start_time = time.time()
        if self.verbose:
            print("EPOCH 0")

    def add_train_loss(self, loss, progress):
        time_elapsed = time.time() - self.start_time
        self.journal[-1]["train"].append({"loss" : loss,
                                          "time" : time_elapsed,
                                          "progress" : progress})
        if self.verbose:
            print(f"{progress:6.2f}% - {time_elapsed:0.0f} - {loss}")

    def add_val_metrics(self, loss, acc=None, bal_acc=None, precision=None,
                        f1=None, silent=False):
        time_elapsed = time.time() - self.start_time
        self.journal[-1]["val"] = {"loss" : loss, "time" : time_elapsed}
        if acc is not None:
            self.journal[-1]["val"]["acc"] = acc
        if bal_acc is not None:
            self.journal[-1]["val"]["bal_acc"] = bal_acc
        if precision is not None:
            self.journal[-1]["val"]["precision"] = precision
        if f1 is not None:
            self.journal[-1]["val"]["f1"] = f1

        if self.verbose and not silent:
            print(f"val metrics ({time_elapsed:0.0f})")
            for metric in self.journal[-1]["val"]:
                print(f"  {metric}: {self.journal[-1]['val'][metric]}")

    def step(self, save=True):
        if self.verbose:
            print(f"EPOCH {len(self.journal)}")
        if save:
            self.save()
        self.journal.append({"train" : []})

    def save(self):
        with open(self.journal_file, "w") as out_f:
            out_f.write(json.dumps(self.journal))

def train(network, bs, lr, epochs, device, full, scheduler_conf, sample_rate,
          id, val_rate, split, weigh_loss, model_name):
    """Function for training ResNet-18 model on CelebA dataset. Should
    be configured using command line arguments (see
    `celeba_resnet_train.py -h` for more information)
    """
    if sample_rate == 1:
        sample_rate = None
    split_f = None
    if split != None:
        split_f = f"{SPLIT_DIR}/split_{split}.txt"
    full_dir = None
    if full:
        full_dir = UNALIGNED_DIR

    dataset = CelebA(CELEBA_DIR, use_transforms=True, normalize=False,
                     sample_rate=sample_rate, custom_splits=split_f,
                     full_dir=full_dir)
    dataloader = DataLoader(dataset, batch_size=bs,num_workers=6,shuffle=True)
    print(f"Training set size: {len(dataset)} samples")

    dataset_val = CelebA(CELEBA_DIR, "val", full_dir=full_dir,
                         normalize=False, custom_splits=split_f)
    dataloader_val = DataLoader(dataset_val, batch_size=bs,
                                num_workers=6, shuffle=True)

    optimizer = optim.SGD(network.parameters(),
                          lr=lr, momentum=0.9, weight_decay=0.0001)

    if scheduler_conf["type"] == "plateau":
        scheduler = ReduceLROnPlateau(optimizer,
            factor=scheduler_conf["factor"],
            patience=scheduler_conf["patience"], verbose=True)
    elif scheduler_conf["type"] == "multiplicative":
        scheduler = MultiplicativeLR(optimizer,
            lr_lambda=lambda epoch: scheduler_conf["multiplier"])
    else:
        raise NotImplementedError("invalid scheduler selection")

    labels = torch.tensor(dataset.labels)
    labels[labels == -1] = 0
    balance = torch.mean(labels.float(), dim=0).to(device)
    pos_weight = (1-balance)/balance
    if weigh_loss:
        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_function = nn.BCEWithLogitsLoss()

    print_iter = min(150, len(dataloader) - 1)
    train_log = TrainLog(journal_file=f"{LOG_DIR}/{model_name}.txt")
    train_log.start_log()
    for epoch in range(epochs):
        avg_loss = 0
        for i, (batch, labels) in enumerate(dataloader):
            batch = batch.to(device)
            labels = labels.float().to(device)

            network.zero_grad()
            output = network.forward(batch)

            loss = loss_function(output, labels)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

            if i % print_iter == 0 and i != 0:
                train_log.add_train_loss(avg_loss/print_iter,
                                         i/len(dataloader)*100)
                avg_loss = 0
                torch.save(network.state_dict(), f"{MODEL_DIR}/{model_name}")
            torch.save(network.state_dict(), f"{MODEL_DIR}/{model_name}")

        with torch.no_grad():
            network.eval()
            avg_loss = 0
            tp = fp = tn = fn = 0
            if epoch % val_rate == 0 or epoch == epochs - 1:
                for batch, labels in dataloader_val:
                    batch = batch.to(device)
                    labels = labels.float().to(device)

                    output = network.forward(batch)
                    loss = loss_function(output, labels)

                    preds = torch.sigmoid(output) > 0.5
                    tp += torch.sum(preds[preds==1] == labels[preds==1]).item()
                    fp += torch.sum(preds[preds==1] != labels[preds==1]).item()
                    tn += torch.sum(preds[preds==0] == labels[preds==0]).item()
                    fn += torch.sum(preds[preds==0] != labels[preds==0]).item()

                    avg_loss += loss.item()

                accuracy = (tp + tn)/(tp + tn + fp + fn)
                precision = tp/(tp + fp)
                recall = tp/(tp + fn)
                f1 = 2*(precision*recall)/(precision + recall)
                balanced_acc = .5*(tp/(tp + fn) + tn/(tn + fp))

                train_log.add_val_metrics(loss=avg_loss/len(dataloader_val),
                    acc=accuracy, bal_acc=balanced_acc, f1=f1)
                if scheduler_conf["type"] == "plateau":
                    scheduler.step(avg_loss)
            else:
                train_log.add_val_metrics(None, silent=True)
            if scheduler_conf["type"] == "multiplicative":
                scheduler.step()
            network.train()
            train_log.step()

def generate_model_name(args):
    model_name = "resnet18_" + args.scheduler
    if not args.aligned:
        model_name += "_unaligned"
    if args.sample_rate != 1:
        model_name += f"_{100//args.sample_rate}p"
    if args.split_idx is not None:
        model_name += f"_split{args.split_idx}"
    if args.weigh_loss:
        model_name += "_weighted"

    id = args.id
    if args.pretrain:
        id = "pretrain_" + id
    if id != "":
        model_name += ("_" + id)

    return model_name

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    Path(LOG_DIR).mkdir(exist_ok=True)
    Path(MODEL_DIR).mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Trains a ResNet-18 model on "
        "the CelebA dataset.")
    parser.add_argument('-b', '--batch-size', type=int, default=256,
        help="Batch size. Default: 256")
    parser.add_argument('-l', '--lr', type=float, default=.1,
        help="Learning rate. Default: .1.")
    parser.add_argument('-e', '--epochs', type=int, default=40,
        help="Number of epochs to train. Default: 40.")
    parser.add_argument('-d', '--device', default="cuda:0",
        help="Device to use for training. Only tested for GPU training."
        " Default: cuda:0.")
    parser.add_argument('-p', '--pretrain', action='store_true', default=False,
        help="Use weights pretrained on imagenet. Default: False.")
    parser.add_argument('-a', '--aligned', action='store_false', default=True,
        help="Use aligned version of CelebA. Default: True.")
    parser.add_argument('-s', '--scheduler', choices=["multiplicative",
        "plateau"], default="multiplicative",
        help="Learning rate scheduler to use. Default: multiplicative.")
    parser.add_argument('--multiplier', type=float, default=.9,
        help="Multiplier to use for multiplicative learning rate schedule. Has"
        " no effect if another scheduler is used. Default: .9.")
    parser.add_argument('--patience', type=int, default=10,
        help="Patience parameter to use for lr reduction on plateau. Has"
        " no effect if another scheduler is used. Default: 10.")
    parser.add_argument('--factor', type=float, default=.1,
        help="Factor parameter to use for lr reduction on plateau. Has"
        " no effect if another scheduler is used. Default: .1.")
    parser.add_argument('-r', '--sample-rate', type=int, default=1,
        help="Rate to downsample the dataset by. Will make a new numpy file "
        "in ./samples/ unless one already exists. Default: 1.")
    parser.add_argument('-c', '--split-idx', type=int, default=None,
        help="Index of custom splits file. Should be located in ./new-splits/"
        ". Default: None.")
    parser.add_argument('-i', '--id', default="",
        help="String to append to model name / log file. Default: None.")
    parser.add_argument('-w', '--weigh-loss', action='store_true',
        help="Weigh attribute losses using ratio between negative and positive"
         " examples in training set. Default: False.",
        default=False)
    parser.add_argument('-v', '--val-rate', type=int, default=1,
        help="If set, will only compute and print vaidation results every "
        "val_rate epochs. Note that the lr reduction on plateau schedule "
        "requires validation results to be computed every epoch.")

    args = parser.parse_args()

    if args.scheduler == "multiplicative":
        scheduler = {"type" : "multiplicative", "multiplier" : args.multiplier}
    elif args.scheduler == "plateau":
        scheduler = {"type" : "plateau", "patience" : args.patience,
                     "factor" : args.factor}
    else:
        raise NotImplementedError("invalid scheduler selection")

    device = torch.device(args.device)

    network = AttributeNN(40, args.pretrain)
    network.to(device)
    print(count_parameters(network), "parameters")

    model_name = generate_model_name(args)
    print(model_name)

    train(network, bs=args.batch_size, lr=args.lr, epochs=args.epochs,
          device=device, full=(not args.aligned), scheduler_conf=scheduler,
          sample_rate=args.sample_rate, id=id, val_rate=args.val_rate,
          split=args.split_idx, weigh_loss=args.weigh_loss,
          model_name=model_name)

if __name__ == "__main__":
    main()
