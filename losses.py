import torch
import torch.nn as nn


class MaskedBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets, mask):
        probs = torch.sigmoid(logits)

        bce = self.bce(logits, targets)
        bce = (bce * mask).sum() / (mask.sum() + 1e-8)

        intersection = (probs * targets * mask).sum()
        union = ((probs + targets) * mask).sum()

        dice = 1 - (2 * intersection + 1e-8) / (union + 1e-8)

        return bce + dice