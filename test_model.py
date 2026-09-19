import torch
from model import load_autoencoder

MODEL_PATH = "optimization2_model.pth"

device = torch.device("cpu")

model = load_autoencoder(
    MODEL_PATH,
    device=device
)

print("Optimized model loaded successfully.")