import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import NeuralNavigatorDataset
from model import NeuralNavigator

def visualize(image, gt_path, pred_path, save_path=None):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.axis("off")

    if gt_path is not None:
        gt = np.array(gt_path)
        plt.plot(gt[:,0], gt[:,1], "g--", label="Ground Truth")

    pred = np.array(pred_path)
    plt.plot(pred[:,0], pred[:,1], "r-", label="Predicted")

    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = NeuralNavigatorDataset("data")
    image, text, gt_path = dataset[0]

    # Load model
    model = NeuralNavigator().to(device)
    model.load_state_dict(torch.load("outputs/neural_navigator.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred_path = model(
            image.unsqueeze(0).to(device),
            text.unsqueeze(0).to(device)
        )

    pred_path = pred_path.squeeze(0).cpu().numpy()
    gt_path = gt_path.numpy()

    # Convert tensor image to displayable format
    img = image.permute(1,2,0).numpy()

    visualize(img, gt_path, pred_path, save_path="outputs/predictions/result.png")

    print("✅ Inference completed. Saved to outputs/predictions/result.png")

if __name__ == "__main__":
    run_inference()
