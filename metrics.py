def dice_score(preds, targets, mask):
    preds = (preds > 0.5).float()
    inter = (preds * targets * mask).sum()
    union = ((preds + targets) * mask).sum()
    return (2 * inter + 1e-8) / (union + 1e-8)


def iou_score(preds, targets, mask):
    preds = (preds > 0.5).float()
    inter = (preds * targets * mask).sum()
    union = ((preds + targets - preds * targets) * mask).sum()
    return (inter + 1e-8) / (union + 1e-8)