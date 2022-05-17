import numpy as np
import torch

dependencies = ["torch", "numpy"]
default_ckpt_url = "https://www.dropbox.com/s/j8ida1d86hy5tm4/waternet_exported_state_dict-daa0ee.pt?dl=1"


def arr2ten_noeinops(arr):
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


def ten2arr_noeinops(ten):
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


def waternet(pretrained=True, device=None):
    """
    Args
    ----
    pretrained: bool
        Load pretrained weights. Defaults to True

    device:
        torch device. Defaults to None

    Returns
    -------
    preprocess: Preprocessing function before inference
    model: WaterNet model
    postprocess: Postprocessing function after inference

    Example usage:
    ```
    import torch
    import cv2

    # Load from torchhub
    preprocess, postprocess, model = torch.hub.load('tnwei/waternet', 'waternet')
    model.eval();

    # Load one image using OpenCV
    im = cv2.imread("example.png")
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Inference -> return numpy array (1, 3, H, W)
    rgb_ten, wb_ten, he_ten, gc_ten = preprocess(rgb_im)
    out_ten = model(rgb_ten, wb_ten, he_ten, gc_ten)
    out_im = postprocess(out_ten)
    ```
    """
    from waternet.data import transform
    from waternet.net import WaterNet

    model = WaterNet()

    if pretrained is True:
        ckpt = torch.hub.load_state_dict_from_url(
            default_ckpt_url,
            progress=False,  # not a pbar but a percentage printout
            check_hash=True,
        )
        model.load_state_dict(ckpt)

    def preprocess(rgb_arr):
        wb, gc, he = transform(rgb_arr)
        rgb_ten = arr2ten_noeinops(rgb_arr)
        wb_ten = arr2ten_noeinops(wb)
        gc_ten = arr2ten_noeinops(gc)
        he_ten = arr2ten_noeinops(he)
        return rgb_ten, wb_ten, he_ten, gc_ten

    def postprocess(model_out):
        return ten2arr_noeinops(model_out)

    return preprocess, postprocess, model.to(device)
