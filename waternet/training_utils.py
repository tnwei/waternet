import torch
import albumentations as A
import cv2
from pathlib import Path
import os
from .data import transform as preprocess_transform
from typing import Optional
import numpy as np


def arr2ten(arr):
    """Converts (N)HWC numpy array into torch Tensor:
    1. Divide by 255
    2. Rearrange dims: HWC -> CHW or NHWC -> NCHW
    """
    ten = torch.from_numpy(arr) / 255
    if len(ten.shape) == 3:
        # ten = rearrange(ten, "h w c -> 1 c h w")
        ten = torch.permute(ten, (2, 0, 1))

    elif len(ten.shape) == 4:
        # ten = rearrange(ten, "n h w c -> n c h w")
        ten = torch.permute(ten, (0, 3, 1, 2))
    return ten


def ten2arr(ten):
    """Convert NCHW torch Tensor into NHWC numpy array:
    1. Multiply by 255, clip and change dtype to unsigned int
    2. Rearrange dims: CHW -> HWC or NCHW -> NHWC
    """
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    if len(arr.shape) == 3:
        # arr = rearrange(arr, "c h w -> h w c")
        arr = np.transpose(arr, (1, 2, 0))
    elif len(arr.shape) == 4:
        # arr = rearrange(arr, "n c h w -> n h w c")
        arr = np.transpose(arr, (0, 2, 3, 1))

    return arr


class UIEBDataset(torch.utils.data.Dataset):
    """UIEBDataset."""

    def __init__(
        self,
        raw_dir,
        ref_dir,
        im_height: Optional[int] = None,
        im_width: Optional[int] = None,
        transform=None,
    ):
        """
        legacy=True to replicate the paper's parameters
        """
        raw_im_fns = sorted([i.name for i in Path(raw_dir).glob("*.png")])
        ref_im_fns = sorted([i.name for i in Path(ref_dir).glob("*.png")])

        assert set(raw_im_fns) == set(ref_im_fns)

        if transform is not None:
            self.transform = transform
        else:
            # No legacy augmentations
            # Paper uses flipping and rotation transforms to obtain 7 augmented versions of data
            # Rotate by 90, 180, 270 degs, hflip, vflip? Not very clear
            # This is as close as it gets without having to go out of my way to reproduce exactly 7 augmented versions
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )

        self.raw_dir = Path(raw_dir)
        self.ref_dir = Path(ref_dir)
        self.im_fns = raw_im_fns
        self.im_height = im_height
        self.im_width = im_width

    def __len__(self):
        return len(self.im_fns)

    def __getitem__(self, idx):
        # Load image
        raw_im = cv2.imread(os.fspath(self.raw_dir / self.im_fns[idx]))
        ref_im = cv2.imread(os.fspath(self.ref_dir / self.im_fns[idx]))

        if (self.im_width is not None) and (self.im_height is not None):
            # Resize accordingly
            raw_im = cv2.resize(raw_im, (self.im_width, self.im_height))
            ref_im = cv2.resize(ref_im, (self.im_width, self.im_height))
        else:
            # Else resize image to be mult of VGG, required by VGG
            im_w, im_h = raw_im.shape[0], raw_im.shape[1]
            vgg_im_w, vgg_im_h = int(im_w / 32) * 32, int(im_h / 32) * 32
            raw_im = cv2.resize(raw_im, (vgg_im_w, vgg_im_h))
            ref_im = cv2.resize(ref_im, (vgg_im_w, vgg_im_h))

        # Convert BGR to RGB for OpenCV
        raw_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2RGB)
        ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=raw_im, mask=ref_im)
            raw_im, ref_im = transformed["image"], transformed["mask"]
        else:
            pass

        # Preprocessing transforms
        wb, gc, he = preprocess_transform(raw_im)

        # Scale to 0 - 1 float, convert to torch Tensor
        raw_ten = arr2ten(raw_im)
        wb_ten = arr2ten(wb)
        gc_ten = arr2ten(gc)
        he_ten = arr2ten(he)
        ref_ten = arr2ten(ref_im)

        # Was gonna make this a tuple until I realized how confused future me would be
        return {
            "raw": raw_ten,
            "wb": wb_ten,
            "gc": gc_ten,
            "he": he_ten,
            "ref": ref_ten,
        }
