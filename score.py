"""
This is literally just train.py adapted for scoring on the UIEB dataset.
"""
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF
from pathlib import Path
from waternet.net import WaterNet
from waternet.training_utils import UIEBDataset
from tqdm import tqdm
from torchmetrics.functional import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
)
import numpy as np
import argparse
import json
from pprint import pprint
from timeit import default_timer as timer

# Training packages: tqdm, albumentations, torchmetrics

TRAIN_METRICS_NAMES = ["mse", "ssim", "psnr", "perceptual_loss", "loss"]
VAL_METRICS_NAMES = ["mse", "ssim", "psnr", "perceptual_loss"]

# Main train and eval fxns ------


def eval_one_epoch(model, val_dataloader, device):
    model.eval()
    epoch_metrics = {i: 0 for i in VAL_METRICS_NAMES}
    minibatches_per_epoch = len(val_dataloader)
    pbar = tqdm(
        enumerate(val_dataloader),
        total=minibatches_per_epoch,
        ascii=True,
        desc="Validation",
        bar_format="{l_bar}{bar:20}{r_bar}",
    )

    with torch.no_grad():
        for _, next_data in pbar:
            rgb_ten = next_data["raw"].to(device)
            wb_ten = next_data["wb"].to(device)
            he_ten = next_data["he"].to(device)
            gc_ten = next_data["gc"].to(device)
            ref_ten = next_data["ref"].to(device)

            # Forward prop
            out = model(rgb_ten, wb_ten, he_ten, gc_ten)

            # Perceptual loss calculation
            imagenet_normalized_x = TF.normalize(
                out, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            imagenet_normalized_y = TF.normalize(
                ref_ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            # size is torch.Size([1, 512, M, N]), where M, N = H/16, W/16
            perceptual_dist = torch.square(
                255
                * (vgg_model(imagenet_normalized_x) - vgg_model(imagenet_normalized_y))
            )
            perceptual_loss = torch.mean(perceptual_dist)

            # Evaluate and record metrics
            epoch_metrics["mse"] += torch.mean(
                torch.square(255 * (out - ref_ten))
            ).item()
            ssim = structural_similarity_index_measure(preds=out, target=ref_ten)
            psnr = peak_signal_noise_ratio(preds=out, target=ref_ten, data_range=1 - 0)
            epoch_metrics["ssim"] += ssim.item()
            epoch_metrics["psnr"] += psnr.item()
            epoch_metrics["perceptual_loss"] = perceptual_loss.item()

    # Update epoch metrics
    epoch_metrics = {i: j / minibatches_per_epoch for i, j in epoch_metrics.items()}

    model.train()
    return epoch_metrics


if __name__ == "__main__":
    start_ts = timer()
    projectroot = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputdir = projectroot / "training"
    torch.manual_seed(0)

    # Config section ------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # Default not specified, so that this argument is blank if unspecified
        "--weights",
        type=str,
        help=f"Weights for scoring",
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="(Optional) Num epochs, defaults to 400"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="(Optional) Batch size, defaults to 16",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=112,
        help="(Optional) Image height, defaults to 112",
    )
    parser.add_argument(
        "--width", type=int, default=112, help="(Optional) Image width, defaults to 112"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(Optional) Seed to pass to `torch.random_seed()` for reproducibility, defaults to None i.e. random",
    )
    args = parser.parse_args()
    assert args.weights is not None, "--weights argument not passed!"

    num_epochs = args.epochs
    batch_size = args.batch_size
    im_height = args.height
    im_width = args.width
    checkpoint_dir = args.weights

    if args.seed is not None:
        torch.manual_seed(args.seed)

    dataset = UIEBDataset(
        projectroot / "data/raw-890",
        projectroot / "data/reference-890",
        im_height=im_height,
        im_width=im_width,
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 90])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Init network ------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = WaterNet()

    if checkpoint_dir is not None:
        with open(checkpoint_dir, "rb") as f:
            model.load_state_dict(torch.load(f, map_location=device))

    model.to(device)
    model.eval()

    # VGG model for perceptual loss
    class PerceptualModel(nn.Module):
        def __init__(self):
            super().__init__()
            vgg_model = torchvision.models.vgg19(pretrained=True)

            # Keep all layers of the backbone except the final maxpool
            self.model = nn.Sequential(*list(vgg_model.features.children())[:-1])

        def forward(self, x):
            return self.model(x)

    vgg_model = PerceptualModel()
    vgg_model.to(device)
    vgg_model.eval()

    # Score models ------

    eval_metrics = eval_one_epoch(model, val_dataloader, device)
    pprint(eval_metrics)
