import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF
from pathlib import Path
from waternet.data import transform
from waternet.net import WaterNet
from waternet.training_utils import UIEBDataset
from tqdm import tqdm

# Training packages: tqdm, albumentations

# Config section ------
# TODO: Replace with OmegaConf for flexibility?
num_epochs = 400
batch_size = 16
im_height = 112
im_width = 112
checkpoint_dir = None

# Main train and eval fxns ------


def eval_one_epoch(model, val_dataloader, device):
    model.eval()
    epoch_metrics = {"mse": 0}
    minibatches_per_epoch = len(val_dataloader)
    pbar = tqdm(
        enumerate(val_dataloader),
        total=minibatches_per_epoch,
        ascii=True,
        bar_format="{l_bar}{bar:10}{r_bar}",
    )

    with torch.no_grad():
        for _, next_data in pbar:
            rgb_ten = next_data["raw"].to(device)
            wb_ten = next_data["wb"].to(device)
            he_ten = next_data["he"].to(device)
            gc_ten = next_data["gc"].to(device)
            ref_ten = next_data["raw"].to(device)

            # Forward prop
            out = model(rgb_ten, wb_ten, he_ten, gc_ten)

            # Evaluate and record metrics
            # TODO: Add the other scoring fxns
            epoch_metrics["mse"] += torch.mean(torch.square(out - ref_ten)).item()

    # Update epoch metrics
    epoch_metrics = {i: j / minibatches_per_epoch for i, j in epoch_metrics.items()}

    # Print epoch metrics
    print(f"MSE: {epoch_metrics['loss']:.2f}")

    model.train()
    return epoch_metrics


def train_one_epoch(model, train_dataloader, optimizer, scheduler, vgg_model, device):
    model.train()
    epoch_metrics = {"loss": 0}
    minibatches_per_epoch = len(train_dataloader)
    pbar = tqdm(
        enumerate(train_dataloader),
        total=minibatches_per_epoch,
        ascii=True,
        bar_format="{l_bar}{bar:10}{r_bar}",
    )

    for idx, next_data in pbar:
        rgb_ten = next_data["raw"].to(device)
        wb_ten = next_data["wb"].to(device)
        he_ten = next_data["he"].to(device)
        gc_ten = next_data["gc"].to(device)
        ref_ten = next_data["raw"].to(device)

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
            vgg_model(imagenet_normalized_x) - vgg_model(imagenet_normalized_y)
        )
        perceptual_loss = torch.mean(perceptual_dist)

        # MSE loss
        mse_loss = torch.mean(torch.square(out - ref_ten))

        # Composite loss
        loss = 0.05 * perceptual_loss + mse_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate and record metrics
        # TODO: Add the other scoring fxns
        epoch_metrics["loss"] += loss.item()

        # Update progress bar
        if (idx % 10 == 0) and (idx != 0):
            pbar.set_postfix({"loss": loss.item()})

    # Update epoch metrics
    epoch_metrics = {i: j / minibatches_per_epoch for i, j in epoch_metrics.items()}
    return epoch_metrics


if __name__ == "__main__":
    # TODO: separate legacy mode for replicating the paper, and the up-to-date implementation
    projectroot = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading section ------
    # TODO: Abstract away in yaml file? Similar to yolov5 so can swap in other datasets
    # TODO: Pre-split the UIEB dataset
    dataset = UIEBDataset(
        projectroot / "data/raw-890", projectroot / "data/reference-890", legacy=True
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 90])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Init network ------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = WaterNet()
    model = model.to(device)

    # TODO: Load weights if available
    if checkpoint_dir is not None:
        with open(checkpoint_dir, "rb") as f:
            model.load_state_dict(torch.load(f, map_location=device))

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
    vgg_model.eval()

    # Actual training loop ------
    for i in range(num_epochs):
        train_one_epoch(
            model, train_dataloader, optimizer, scheduler, vgg_model, device
        )
        eval_one_epoch(model, val_dataloader, device)  # will print epoch metrics

        # TODO
        torch.save(model.state_dict(), "last.pt")
