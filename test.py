import torch
from metrics import dice_score, iou_score

from settings import settings


def evaluate(model, test_loader):
    model.eval()

    dices, ious = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(settings.DEVICE)
            y = batch['y'].to(settings.DEVICE)
            mask = batch['mask_map'].to(settings.DEVICE)

            probs = torch.sigmoid(model(x))

            dices.append(dice_score(probs, y, mask).item())
            ious.append(iou_score(probs, y, mask).item())

    print("TEST:")
    print("Dice:", sum(dices) / len(dices))
    print("IoU:", sum(ious) / len(ious))