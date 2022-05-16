import torch
import torch.nn as nn

# import torch.nn.functional as F


class ConfidenceMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Confidence maps
        # Accepts input of size (N, 3*4, H, W)
        self.conv1 = nn.Conv2d(
            in_channels=12, out_channels=128, kernel_size=7, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=5, dilation=1, padding="same"
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, dilation=1, padding="same"
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=1, dilation=1, padding="same"
        )
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, dilation=1, padding="same"
        )
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, dilation=1, padding="same"
        )
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding="same"
        )
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, wb, ce, gc):
        out = torch.cat([x, wb, ce, gc], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.relu7(self.conv7(out))
        out = self.sigmoid(self.conv8(out))
        out1, out2, out3 = torch.split(out, [1, 1, 1], dim=1)
        return out1, out2, out3


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=32, kernel_size=7, dilation=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, dilation=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, dilation=1, padding="same"
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, xbar):
        out = torch.cat([x, xbar], dim=1)
        out = self.relu1(self.conv1(out))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        return out


class WaterNet(nn.Module):
    """
    waternet = WaterNet()
    in = torch.randn(16, 3, 112, 112)
    waternet_out = waternet(in, in, in, in)
    waternet_out.shape
    # torch.Size([16, 3, 112, 112])
    """

    def __init__(self):
        super().__init__()
        self.cmg = ConfidenceMapGenerator()
        self.wb_refiner = Refiner()
        self.ce_refiner = Refiner()
        self.gc_refiner = Refiner()

    def forward(self, x, wb, ce, gc):
        wb_cm, ce_cm, gc_cm = self.cmg(x, wb, ce, gc)
        refined_wb = self.wb_refiner(x, wb)
        refined_ce = self.ce_refiner(x, ce)
        refined_gc = self.gc_refiner(x, gc)
        return (
            torch.mul(refined_wb, wb_cm)
            + torch.mul(refined_ce, ce_cm)
            + torch.mul(refined_gc, gc_cm)
        )
