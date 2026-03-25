import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from dataset import GeologyTrapsDataset
from utils.data_utils import get_all_files
from model import get_model
from losses import MaskedBCEDiceLoss
from settings import settings


def plot_loss_curve(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Sanity Check Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def visualize_batch(model, batch):
    model.eval()

    x = batch['x']
    y = batch['y']
    mask = batch['mask_map']

    with torch.no_grad():
        logits = model(x.to(settings.DEVICE))
        probs = torch.sigmoid(logits).cpu()

    rgb = x[0][:3].permute(1, 2, 0).numpy()
    depth = x[0][3].numpy()
    gt = y[0][0].numpy()
    pred = probs[0][0].numpy()
    mask_map = mask[0][0].numpy()

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    axs[0].imshow(rgb)
    axs[0].set_title("RGB")

    axs[1].imshow(depth, cmap='gray')
    axs[1].set_title("Depth")

    axs[2].imshow(gt, cmap='gray')
    axs[2].set_title("GT traps")

    axs[3].imshow(pred, cmap='jet')
    axs[3].set_title("Prediction")

    axs[4].imshow(rgb)
    axs[4].imshow(pred, alpha=0.4, cmap='jet')
    axs[4].set_title("Overlay")

    for ax in axs:
        ax.axis('off')

    plt.show()


def sanity_check(num_samples=2, epochs=150):
    print("=== SANITY CHECK START ===")

    # ==== DATA ====
    files = get_all_files()

    dataset = GeologyTrapsDataset(
        files,
        augment=False,              # ❗ без аугментаций
        use_faults=settings.USE_FAULTS,
        data_source=settings.DATA_SOURCE
    )

    assert len(dataset) > num_samples, "Слишком мало данных"

    subset = Subset(dataset, list(range(num_samples)))
    loader = DataLoader(subset, batch_size=1, shuffle=True)

    print(f"Using {num_samples} samples")

    # ==== MODEL ====
    model = get_model().to(settings.DEVICE)

    # ❗ высокий LR для быстрого переобучения
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = MaskedBCEDiceLoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in loader:
            x = batch['x'].to(settings.DEVICE)
            y = batch['y'].to(settings.DEVICE)
            mask = batch['mask_map'].to(settings.DEVICE)

            logits = model(x)
            loss = criterion(logits, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={epoch_loss:.4f}")
            visualize_batch(model, batch)

    plot_loss_curve(losses)

    print("=== SANITY CHECK DONE ===")

    return model