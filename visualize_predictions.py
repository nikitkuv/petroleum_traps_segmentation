import matplotlib.pyplot as plt
import torch


def visualize_predictions(model, loader, num_samples=5):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch['x']
            logits = model(x.to(next(model.parameters()).device))
            probs = torch.sigmoid(logits).cpu()

            for i in range(x.size(0)):
                if count >= num_samples:
                    return

                rgb = x[i][:3].permute(1, 2, 0).numpy()
                pred = probs[i][0].numpy()

                plt.figure(figsize=(6, 10))
                plt.imshow(rgb)
                plt.imshow(pred, alpha=0.4, cmap='jet')
                plt.title("Prediction overlay")
                plt.axis('off')
                plt.show()

                count += 1