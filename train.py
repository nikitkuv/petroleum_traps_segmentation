import torch
import wandb
from tqdm import tqdm
from dataloaders import create_dataloaders
from model import get_model
from losses import MaskedBCEDiceLoss
from metrics import dice_score, iou_score
from optim import get_optimizer, get_scheduler

from settings import settings


def train():
    wandb.init(project="petroleum-traps", config=vars(settings))

    train_loader, val_loader, test_loader = create_dataloaders()

    model = get_model().to(settings.DEVICE)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = MaskedBCEDiceLoss()

    best_dice = 0

    for epoch in range(settings.NUM_EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            x = batch['x'].to(settings.DEVICE)
            y = batch['y'].to(settings.DEVICE)
            mask = batch['mask_map'].to(settings.DEVICE)

            logits = model(x)
            loss = criterion(logits, y, mask)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== VAL =====
        model.eval()
        val_loss, dices, ious = 0, [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(settings.DEVICE)
                y = batch['y'].to(settings.DEVICE)
                mask = batch['mask_map'].to(settings.DEVICE)

                logits = model(x)
                loss = criterion(logits, y, mask)

                probs = torch.sigmoid(logits)

                dices.append(dice_score(probs, y, mask).item())
                ious.append(iou_score(probs, y, mask).item())

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_dice = sum(dices) / len(dices)
        val_iou = sum(ious) / len(ious)

        scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "grad_norm": grad_norm.item()
        })

        print(f"Epoch {epoch}: Dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), settings.checkpoint_path / "best_model.pth")

    return model, test_loader