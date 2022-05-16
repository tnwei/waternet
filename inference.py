import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange

from waternet.data import transform
from waternet.net import WaterNet

# Config ------

wd = Path(__file__).parent.resolve()  # repo root
outputdir = wd / "output"
default_ckpt_dir_relative = "waternet-exported-state-dict.pt"
default_ckpt_dir_absolute = wd / default_ckpt_dir_relative

# Dropbox URLs just need dl=1 to ensure direct download link
default_ckpt_url = (
    "https://www.dropbox.com/s/j8ida1d86hy5tm4/waternet-exported-state-dict.pt?dl=1"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Util fxns ------


def arr2ten(arr):
    """Convert arr2ten plus scaling"""
    ten = torch.from_numpy(arr) / 255
    ten = rearrange(ten, "h w c -> 1 c h w")
    return ten


def ten2arr(ten):
    """Convert ten2arr plus scaling"""
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = rearrange(arr, "c h w -> h w c")
    return arr


# Parse args ------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    type=str,
    help="Path to input image/video, any format works as long as OpenCV can open it.",
)
parser.add_argument(
    # Default not specified, so that this argument is blank if unspecified
    "--weights",
    type=str,
    help=f"(Optional) Path to model weights, defaults to {default_ckpt_dir_relative}. Auto-downloads pretrained weights if not available.",
)
parser.add_argument(
    "--name", type=str, help="(Optional) Subfolder name to save under `./output`."
)
args = parser.parse_args()

assert args.source is not None, "No input image/video specified in --source!"

# Load weights ------

model = WaterNet()
model = model.to(device)

if args.weights is None:
    print(
        f"No weights specified in --weights, using default: {default_ckpt_dir_relative}"
    )
    weights_dir = default_ckpt_dir_absolute

    if not weights_dir.exists():
        # Need to download weights
        # Redundant, torch has its own printout
        # print(
        #     "Downloading pretrained weights: https://www.dropbox.com/s/j8ida1d86hy5tm4/waternet-exported-state-dict.pt"
        # )
        sd = torch.hub.load_state_dict_from_url(
            default_ckpt_url,
            progress=False,  # not a pbar but a percentage printout
            map_location=device,
            model_dir=wd,
        )
        # print(f"Pretrained weights saved to {weights_dir}") # Redundant
        model.load_state_dict(sd)
else:
    weights_dir = args.weights

    with open(weights_dir, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))


# Load source, figure out source type ------

assert Path(args.source).exists(), f"{args.source} does not exist!"

if not Path(args.source).is_file():
    raise ValueError(f"{args.source} is not a file, but a folder")
else:
    im = cv2.imread(args.source)

    if im is None:
        is_video = True
    else:
        is_video = False

    if is_video is True:
        # Load as video
        raise NotImplementedError("Video not implemented yet")
    elif is_video is False:
        # Load as image
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Run preprocessing + inference on provided input ------
if is_video is False:
    wb, gc, he = transform(im)
    rgb_ten = arr2ten(im)
    wb_ten = arr2ten(wb)
    gc_ten = arr2ten(gc)
    he_ten = arr2ten(he)

    with torch.no_grad():
        rgb_ten = rgb_ten.to(device)
        wb_ten = wb_ten.to(device)
        gc_ten = gc_ten.to(device)
        he_ten = he_ten.to(device)

        # torch.Size([1, 3, H, W])
        out = model(rgb_ten, wb_ten, he_ten, gc_ten)
        out_im = ten2arr(out[0])
        out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)

elif is_video is True:
    raise NotImplementedError("Video not implemented yet")

# Figure out savedir ------
# Implemented towards the back, prevent runtime errors creating empty folders

# Create outputdir if not exists
if not outputdir.exists():
    outputdir.mkdir()

# Determine savedir if --name not provided
if args.name is None:
    numerical_subdirs = list(outputdir.glob("*"))

    # isdecimal over isdigit and isnumeric
    # see: https://datagy.io/python-isdigit/
    numerical_subdirs = [
        int(i.stem) for i in numerical_subdirs if (i.is_dir() and i.stem.isdecimal())
    ]

    if len(numerical_subdirs) == 0:
        savedir = outputdir / "0"
        savedir.mkdir()
    else:
        savedir = outputdir / str(max(numerical_subdirs) + 1)
        savedir.mkdir()
else:
    savedir = outputdir / args.name
    if not savedir.exists():
        savedir.mkdir()


# Save files
if is_video is False:
    og_filename = Path(args.source).name
    outpath = (savedir / og_filename).as_posix()
    cv2.imwrite(outpath, out_im)
    print(f"Saved to {outpath}!")
else:
    raise NotImplementedError("Video not implemented yet")
