import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcelogits = nn.BCEWithLogitsLoss()

    def forward(self, pred, mask, smooth=1):
        mask = mask.to(dtype=torch.float32)

        bce_loss = self.bcelogits(pred, mask)

        pred = F.sigmoid(pred)
        pred = pred.view(-1)
        mask = mask.view(-1)

        intersection = (pred * mask).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth)

        return bce_loss + (1 - dice)
