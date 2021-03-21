import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from celeba import CelebA
from celeba_resnet_train import AttributeNN
import torchvision.models as models
import argparse

DEFAULT_CELEBA_DIR = "./CelebA"
DEFAULT_UNALIGNED_DIR = "./img_celeba"

def evaluate(network, dataloader, device):
    tp = torch.zeros(40).to(device)
    tn = torch.zeros(40).to(device)
    fp = torch.zeros(40).to(device)
    fn = torch.zeros(40).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    for batch, labels in dataloader:
        batch = batch.to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            output = network.forward(batch)
            loss = loss_function(output, labels)
            preds = (torch.sigmoid(output) > 0.5).type(torch.int)

        tp += torch.sum(preds + labels == 2, axis=0).type(torch.float)
        fp += torch.sum(preds - labels == 1, axis=0).type(torch.float)
        tn += torch.sum(preds + labels == 0, axis=0).type(torch.float)
        fn += torch.sum(preds - labels == -1, axis=0).type(torch.float)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*(precision*recall)/(precision + recall)
    balanced_acc = .5*(tp/(tp + fn) + tn/(tn + fp))

    avg_accuracy = (tp.sum()+tn.sum())/(tp.sum()+tn.sum()+fp.sum()+fn.sum())
    avg_precision = tp.sum()/(tp.sum() + fp.sum())
    avg_recall = tp.sum()/(tp.sum() + fn.sum())
    avg_f1 = 2*(avg_precision*avg_recall)/(avg_precision + avg_recall)
    avg_balanced_acc = .5*(tp.sum()/(tp.sum() + fn.sum())
                           + tn.sum()/(tn.sum() + fp.sum()))

    return {"acc" : accuracy, "bal_acc" : balanced_acc, "f1" : f1,
            "precision" : precision, "recall" : recall,
            "avg_acc" : avg_accuracy.item(),
            "avg_bal_acc" : avg_balanced_acc.item(), "avg_f1" : avg_f1.item(),
            "avg_precision" : avg_precision.item(),
            "avg_recall" : avg_recall.item()}

def main():
    parser = argparse.ArgumentParser(description="Calculates metrics for a "
        "trained ResNet-18 model on the CelebA dataset.")
    parser.add_argument('model_path', help="Path to model weights.")
    parser.add_argument('-b', '--batch-size', type=int, default=256,
        help="Batch size. Default: 256")
    parser.add_argument('-a', '--aligned', action='store_false', default=True,
        help="Use aligned version of CelebA. Default: True.")
    parser.add_argument('-c', '--celeba-dir', default=DEFAULT_CELEBA_DIR,
        help=f"Path to CelebA folder. Default: {DEFAULT_CELEBA_DIR}.")
    parser.add_argument('-u', '--unaligned-dir', default=DEFAULT_UNALIGNED_DIR,
        help="Path to img_celeba folder containing unaligned images. "
        f"Default: {DEFAULT_UNALIGNED_DIR}.")
    parser.add_argument('-s', '--split', choices=["train","val","test","full"],
        help="Split to compute metrics on. Default: val.", default="val")
    parser.add_argument('-d', '--device', default="cuda:0",
        help="Device to use for evaluation. Default: cuda:0.")
    parser.add_argument('--average', action='store_true', default=False,
        help="Combine 3 iterations of a model. Default: False.")

    args = parser.parse_args()

    if args.aligned:
        dataset = CelebA(args.celeba_dir, args.split, normalize=False)
    else:
        dataset = CelebA(args.celeba_dir, args.split,
                         full_dir=args.unaligned_dir, normalize=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=6)

    network = AttributeNN(40)
    network.to(args.device)
    network.eval()

    metric_names = ["acc", "bal_acc", "f1", "precision", "recall"]
    if args.average:
        metrics = {}
        results = []
        for i in range(5):
            network.load_state_dict(torch.load(f"{args.model_path}_{i+1}",
                map_location=args.device))
            results.append(evaluate(network, dataloader, args.device))
        for name in metric_names:
            metrics[name] = torch.mean(torch.cat(
                [result[name] for result in results]).view(-1,40),dim=0)
            metrics["avg_" + name] = sum(
                [result["avg_" + name] for result in results])/5

            avg_metrics = torch.mean(torch.cat(
                [result[name] for result in results]).view(-1,40),dim=1)
            metrics["attr_avg_" + name] = torch.mean(avg_metrics)
            metrics["attr_std_" + name] = torch.std(avg_metrics,unbiased=False)
    else:
        network.load_state_dict(torch.load(args.model_path,
            map_location=args.device))
        metrics = evaluate(network, dataloader, args.device)
        for name in metric_names:
            metrics["attr_avg_" + name] = torch.mean(metrics[name])

    print("acc    bal_ac f1     prc    rcl")
    for i, attr in enumerate(dataset.attributes):
        print(f"{metrics['acc'][i].item()*100:.3f} "
              f"{metrics['bal_acc'][i].item()*100:.3f} "
              f"{metrics['f1'][i].item()*100:.3f} "
              f"{metrics['precision'][i].item()*100:.3f} "
              f"{metrics['recall'][i].item()*100:.3f} | {attr}")
    print("-"*56)
    print(f"{metrics['avg_acc']*100:.3f} "
          f"{metrics['avg_bal_acc']*100:.3f} "
          f"{metrics['avg_f1']*100:.3f} "
          f"{metrics['avg_precision']*100:.3f} "
          f"{metrics['avg_recall']*100:.3f} | cumulative avg")
    print("-"*56)
    print(f"{metrics['attr_avg_acc'].item()*100:.3f} "
          f"{metrics['attr_avg_bal_acc'].item()*100:.3f} "
          f"{metrics['attr_avg_f1'].item()*100:.3f} "
          f"{metrics['attr_avg_precision'].item()*100:.3f} "
          f"{metrics['attr_avg_recall'].item()*100:.3f} / attr avg mean")
    print("-"*56)
    if args.average:
        print(f"{metrics['attr_std_acc'].item()*100:.4f} "
              f"{metrics['attr_std_bal_acc'].item()*100:.4f} "
              f"{metrics['attr_std_f1'].item()*100:.4f} "
              f"{metrics['attr_std_precision'].item()*100:.4f} "
              f"{metrics['attr_std_recall'].item()*100:.4f} / attr avg std")
        print("-"*56)

if __name__ == "__main__":
    main()
