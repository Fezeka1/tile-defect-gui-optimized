"""
Autoencoder architecture for tile/glass surface defect detection.
 
Encoder: ResNet18 (ImageNet-pretrained) truncated after layer2 -> stride 8,
128 channels. Frozen by default (transfer learning).
Decoder: 3x (ConvTranspose2d -> BatchNorm -> ReLU) upsampling blocks back
to full resolution, + final Conv2d -> Sigmoid to produce an RGB reconstruction
in [0, 1].
 
This matches the state_dict keys in best_baseline_autoencoder.pth exactly
(encoder.features.{0..5}, decoder.net.{0..10}), so `load_state_dict` works
with no renaming.
"""
import torch
import torch.nn as nn
import torchvision.models as models
 
 
class Encoder(nn.Module):
    """Pretrained ResNet18 truncated after layer2, frozen by default."""
 
    def __init__(self, freeze: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # stem + layer1 + layer2 -> output stride 8, 128 channels
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.freeze = freeze
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
 
    def forward(self, x):
        return self.features(x)  # [B, 128, H/8, W/8]
 
 
class Decoder(nn.Module):
    """Trainable decoder: upsamples the encoder's feature map back to a
    reconstructed RGB image at the original resolution."""
 
    def __init__(self, in_channels: int = 128, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
 
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
 
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
 
    def forward(self, z):
        return self.net(z)
 
 
class AutoencoderAD(nn.Module):
    def __init__(self, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = Encoder(freeze=freeze_encoder)
        self.decoder = Decoder(in_channels=128, out_channels=3)
 
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
 
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
 
    def train(self, mode: bool = True):
        """Override so encoder always stays in eval() mode (frozen BN
        statistics) even when the whole model is set to .train()."""
        super().train(mode)
        if self.encoder.freeze:
            self.encoder.eval()
        return self
 
 
def load_autoencoder(checkpoint_path: str, device: str = "cpu", freeze_encoder: bool = True) -> AutoencoderAD:
    """Load a trained AutoencoderAD from a state_dict checkpoint."""
    model = AutoencoderAD(freeze_encoder=freeze_encoder)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
