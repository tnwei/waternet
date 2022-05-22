import argparse
from pathlib import Path
import os

import cv2
import numpy as np
import torch
from waternet.data import transform
from waternet.net import WaterNet

# Config ------

wd = Path(__file__).parent.resolve()  # repo root
outputdir = wd / "output"
default_ckpt_dir_relative = "waternet_exported_state_dict-daa0ee.pt"
default_ckpt_dir_absolute = wd / default_ckpt_dir_relative
VID_SUFFIXES = [".mp4", ".mpeg", ".avi"]
IM_SUFFIXES = [".bmp", ".jpg", ".jpeg", ".png", ".gif"]

# Dropbox URLs just need dl=1 to ensure direct download link
default_ckpt_url = "https://www.dropbox.com/s/j8ida1d86hy5tm4/waternet_exported_state_dict-daa0ee.pt?dl=1"

# Util fxns ------


def arr2ten(arr):
    """Converts (N)HWC numpy array into torch Tensor:
    1. Divide by 255
    2. Rearrange dims: HWC -> 1CHW or NHWC -> NCHW
    """
    ten = torch.from_numpy(arr) / 255
    if len(ten.shape) == 3:
        # ten = rearrange(ten, "h w c -> 1 c h w")
        ten = torch.permute(ten, (2, 0, 1))
        ten = torch.unsqueeze(ten, dim=0)
    elif len(ten.shape) == 4:
        # ten = rearrange(ten, "n h w c -> n c h w")
        ten = torch.permute(ten, (0, 3, 1, 2))
    return ten


def ten2arr(ten):
    """Convert NCHW torch Tensor into NHWC numpy array:
    1. Multiply by 255, clip and change dtype to unsigned int
    2. Rearrange dims: NCHW -> NHWC
    """
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    # arr = rearrange(arr, "n c h w -> n h w c")
    arr = np.transpose(arr, (0, 2, 3, 1))
    return arr


# Parse args ------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    type=str,
    help="Path to input image/video/directory, supports image formats: bmp, jpg, jpeg, png, gif, and video formats: mp4, mpeg, avi",
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

# this becomes `args.show_split`
parser.add_argument(
    "--show-split",
    action="store_true",
    default=False,
    help="(Optional) Left/right of output is original/processed. Adds before/after watermark.",
)
args = parser.parse_args()

assert args.source is not None, "No input image/video specified in --source!"

# Load weights ------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
            check_hash=True,
        )
        # print(f"Pretrained weights saved to {weights_dir}") # Redundant
        model.load_state_dict(sd)
    else:
        with open(weights_dir, "rb") as f:
            model.load_state_dict(torch.load(f, map_location=device))

else:
    weights_dir = args.weights

    with open(weights_dir, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))

model.eval()

# Load source  ------

source_fp = Path(args.source)
assert source_fp.exists(), f"{args.source} does not exist!"

if source_fp.is_dir():
    fdirs = list(source_fp.glob("*"))
    fdirs = [
        i
        for i in fdirs
        if (i.suffix.lower() in VID_SUFFIXES) or (i.suffix.lower() in IM_SUFFIXES)
    ]
else:
    fdirs = [source_fp]

print(f"Total images/videos: {len(fdirs)}")

# Figure out savedir ------

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
    else:
        savedir = outputdir / str(max(numerical_subdirs) + 1)
else:
    savedir = outputdir / args.name


# Preprocessing / inference / saving ------
for fdir in fdirs:
    if fdir.suffix in IM_SUFFIXES:
        # Load image
        bgr = cv2.imread(os.fspath(fdir))  # OpenCV can't read pathlike-objects

        if len(bgr.shape) == 3:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif len(bgr.shape) == 4:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)

        # Preprocessing
        wb, gc, he = transform(rgb)
        rgb_ten = arr2ten(rgb)
        wb_ten = arr2ten(wb)
        gc_ten = arr2ten(gc)
        he_ten = arr2ten(he)

        # Inference + postprocessing
        with torch.no_grad():
            rgb_ten = rgb_ten.to(device)
            wb_ten = wb_ten.to(device)
            gc_ten = gc_ten.to(device)
            he_ten = he_ten.to(device)

            # torch.Size([1, 3, H, W])
            out = model(rgb_ten, wb_ten, he_ten, gc_ten)
            out_im = ten2arr(out)[0]
            out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)

        outpath = os.fspath(savedir / fdir.name)

        # Savedir exists check as late as possible
        # so early errors don't create empty savedirs
        if not savedir.exists():
            savedir.mkdir()

        if args.show_split is True:
            composite_im = np.zeros_like(rgb)
            w = int(out_im.shape[1] / 2)  # h, w, c
            composite_im[:, :w, :] = bgr[:, :w, :]
            composite_im[:, w:, :] = out_im[:, w:, :]

            cv2.putText(
                img=composite_im,
                text=f"Before",
                # location of bottom-left corner of text
                org=(50, 50),  # W, H
                # just pick sth not ugly
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,  # multiplied by font base size
                color=(255, 255, 255),
                thickness=2,
            )

            cv2.putText(
                img=composite_im,
                text=f"After",
                # location of bottom-left corner of text
                org=(w + 50, 50),
                # just pick sth not ugly
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,  # multiplied by font base size
                color=(255, 255, 255),
                thickness=2,
            )
            cv2.imwrite(outpath, composite_im)
        else:
            cv2.imwrite(outpath, out_im)

    elif fdir.suffix in VID_SUFFIXES:
        # Load as video
        # Set up I/O
        cap = cv2.VideoCapture(os.fspath(fdir))
        frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        outpath = os.fspath(savedir / (fdir.stem + ".mp4"))

        print(f"{frame_width=}, {frame_height=}")

        # Savedir exists check as late as possible
        # so early errors don't create empty savedirs
        # Declaring this after init VideoWriter = no video saved!
        if not savedir.exists():
            savedir.mkdir()

        codec = cv2.VideoWriter.fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            outpath, codec, frames_per_second, (frame_width, frame_height)
        )
        print(f"Working on {fdir.name} with {total_frames} frames")

        frames = 0

        while True:
            retval, bgr = cap.read()

            if retval is False:
                break

            # Preprocessing
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            wb, gc, he = transform(rgb)
            rgb_ten = arr2ten(rgb)
            wb_ten = arr2ten(wb)
            gc_ten = arr2ten(gc)
            he_ten = arr2ten(he)

            # Inference + postprocessing
            with torch.no_grad():
                rgb_ten = rgb_ten.to(device)
                wb_ten = wb_ten.to(device)
                gc_ten = gc_ten.to(device)
                he_ten = he_ten.to(device)

                # torch.Size([1, 3, H, W])
                out = model(rgb_ten, wb_ten, he_ten, gc_ten)
                out_im = ten2arr(out)[0]
                out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)

            if args.show_split is True:
                composite_im = np.zeros_like(bgr)
                w = int(out_im.shape[1] / 2)  # h, w, c
                composite_im[:, :w, :] = bgr[:, :w, :]
                composite_im[:, w:, :] = out_im[:, w:, :]

                cv2.putText(
                    img=composite_im,
                    text=f"Before",
                    # location of bottom-left corner of text
                    org=(50, 50),  # W, H
                    # just pick sth not ugly
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,  # multiplied by font base size
                    color=(255, 255, 255),
                    thickness=2,
                )

                cv2.putText(
                    img=composite_im,
                    text=f"After",
                    # location of bottom-left corner of text
                    org=(w + 50, 50),
                    # just pick sth not ugly
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,  # multiplied by font base size
                    color=(255, 255, 255),
                    thickness=2,
                )
                video_writer.write(composite_im)
            else:
                video_writer.write(out_im)

            frames += 1

            if frames % 50 == 0:
                print(f"Processed {frames} frames")

        cap.release()
        video_writer.release()

        cv2.destroyAllWindows()


print(f"Saved output to {savedir}!")
