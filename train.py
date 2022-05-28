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


def train_one_epoch(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    vgg_model,
    device,
    epoch_num,
    total_epochs,
):
    model.train()
    epoch_metrics = {i: 0 for i in TRAIN_METRICS_NAMES}
    minibatches_per_epoch = len(train_dataloader)
    pbar = tqdm(
        enumerate(train_dataloader),
        total=minibatches_per_epoch,
        ascii=True,
        desc=f"Epoch {epoch_num+1}/{total_epochs}",
        bar_format="{l_bar}{bar:20}{r_bar}",
    )
    for idx, next_data in pbar:
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
            255 * (vgg_model(imagenet_normalized_x) - vgg_model(imagenet_normalized_y))
        )
        perceptual_loss = torch.mean(perceptual_dist)

        # MSE loss
        mse = torch.mean(torch.square(255 * (out - ref_ten)))

        # Composite loss
        loss = (0.05 * perceptual_loss) + mse

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate and record metrics
        epoch_metrics["loss"] += loss.item()
        epoch_metrics["perceptual_loss"] += perceptual_loss.item()
        epoch_metrics["mse"] += mse.item()

        with torch.no_grad():
            ssim = structural_similarity_index_measure(preds=out, target=ref_ten)
            psnr = peak_signal_noise_ratio(preds=out, target=ref_ten, data_range=1 - 0)
            epoch_metrics["ssim"] += ssim.item()
            epoch_metrics["psnr"] += psnr.item()

        # Update progress bar
        if (idx % 10 == 0) and (idx != 0):
            pbar.set_postfix({"loss": loss.item()})

    # Update epoch metrics
    epoch_metrics = {i: j / minibatches_per_epoch for i, j in epoch_metrics.items()}
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
        # Default not specified, so that this argument is blank if unspecified
        "--weights",
        type=str,
        help=f"(Optional) Starting weights for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(Optional) Seed to pass to `torch.random_seed()` for reproducibility, defaults to None i.e. random",
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    im_height = args.height
    im_width = args.width
    checkpoint_dir = args.weights

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create outputdir if not exists
    if not outputdir.exists():
        outputdir.mkdir()

    # Determine savedir
    numerical_subdirs = list(outputdir.glob("*"))

    # isdecimal over isdigit and isnumeric
    # see: https://datagy.io/python-isdigit/
    numerical_subdirs = [
        int(i.stem) for i in numerical_subdirs if (i.is_dir() and i.stem.isdecimal())
    ]

    if len(numerical_subdirs) == 0:
        savedir = outputdir / "0"
    else:
        savedir = outputdir / str(max(numerical_subdirs) + 1)

    # Data loading section ------
    # TODO: Abstract away in yaml file? Similar to yolov5 so can swap in other datasets
    # TODO: Pre-split the UIEB dataset
    # Same split everytime using `torch.random_seed()`
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
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

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

    # Main training loop ------
    saved_train_metrics = {i: [] for i in TRAIN_METRICS_NAMES}
    saved_val_metrics = {i: [] for i in VAL_METRICS_NAMES}

    for i in range(num_epochs):
        train_metrics = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            vgg_model,
            device,
            epoch_num=i,
            total_epochs=num_epochs,
        )

        eval_metrics = eval_one_epoch(model, val_dataloader, device)

        print(
            "    Train ||",
            "   ".join([f"{i}: {j:.03g}" for i, j in train_metrics.items()]),
        )
        print(
            "    Val   ||",
            "   ".join([f"{i}: {j:.03g}" for i, j in eval_metrics.items()]),
        )
        print()

        for i, j in train_metrics.items():
            saved_train_metrics[i].append(j)

        for i, j in eval_metrics.items():
            saved_val_metrics[i].append(j)

        # Savedir exists check as late as possible
        # so early errors don't create empty savedirs
        if not savedir.exists():
            savedir.mkdir()

        torch.save(model.state_dict(), savedir / "last.pt")

    # Save train metrics
    train_metrics_arr = np.concatenate(
        [np.array(saved_train_metrics[i]).reshape(-1, 1) for i in TRAIN_METRICS_NAMES],
        axis=1,
    )
    val_metrics_arr = np.concatenate(
        [np.array(saved_val_metrics[i]).reshape(-1, 1) for i in VAL_METRICS_NAMES],
        axis=1,
    )

    np.savetxt(
        savedir / "metrics-train.csv",
        train_metrics_arr,
        fmt="%f",
        delimiter=",",
        comments="",
        header=",".join(TRAIN_METRICS_NAMES),
    )
    np.savetxt(
        savedir / "metrics-val.csv",
        val_metrics_arr,
        fmt="%f",
        delimiter=",",
        comments="",
        header=",".join(VAL_METRICS_NAMES),
    )

    with open(savedir / "config.json", "w") as f:
        json.dump(
            {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "im_height": im_height,
                "im_width": im_width,
                "weights": checkpoint_dir,
            },
            f,
            indent=4,
        )

    print(f"Metrics and weights saved to {savedir}")

    print(f"Total time: {timer()-start_ts}s")
