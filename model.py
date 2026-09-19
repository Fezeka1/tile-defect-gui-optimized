"""
Autoencoder model for tile defect anomaly detection.

Encoder:
    ImageNet-pretrained ResNet18 truncated after layer2.

Decoder:
    Upsamples encoder features back to an RGB reconstruction.

Supports checkpoint decoder naming variants:
    decoder.net.*
    decoder.decoder.*
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        freeze: bool = True
    ):
        super().__init__()


        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )


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

            for parameter in (
                self.features.parameters()
            ):

                parameter.requires_grad = False


    def forward(self, x):

        return self.features(x)


# ============================================================
# DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3
    ):
        super().__init__()


        self.net = nn.Sequential(

            # H/8 -> H/4
            nn.ConvTranspose2d(
                in_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(
                inplace=True
            ),


            # H/4 -> H/2
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(
                inplace=True
            ),


            # H/2 -> H
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(
                inplace=True
            ),


            # RGB output
            nn.Conv2d(
                16,
                out_channels,
                kernel_size=3,
                padding=1
            ),


            nn.Sigmoid(),
        )


    def forward(self, z):

        return self.net(z)


# ============================================================
# AUTOENCODER
# ============================================================

class AutoencoderAD(nn.Module):

    def __init__(
        self,
        freeze_encoder: bool = True
    ):
        super().__init__()


        self.encoder = Encoder(
            freeze=freeze_encoder
        )


        self.decoder = Decoder(
            in_channels=128,
            out_channels=3
        )


    def forward(self, x):

        encoded = self.encoder(x)

        reconstruction = self.decoder(
            encoded
        )

        return reconstruction


    def trainable_parameters(self):

        return [
            parameter

            for parameter
            in self.parameters()

            if parameter.requires_grad
        ]


    def train(
        self,
        mode: bool = True
    ):

        super().train(mode)


        if self.encoder.freeze:

            self.encoder.eval()


        return self


# ============================================================
# MODEL LOADER
# ============================================================

def load_autoencoder(
    checkpoint_path: str,
    device="cpu",
    freeze_encoder: bool = True
):
    """
    Load AutoencoderAD checkpoint.

    Supports:

        raw state_dict

    and:

        {
            "model_state_dict": ...
        }

    Also supports decoder naming:

        decoder.net.*

    and:

        decoder.decoder.*
    """


    model = AutoencoderAD(
        freeze_encoder=freeze_encoder
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )


    # --------------------------------------------------------
    # Extract model weights
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Decoder-name compatibility
    # --------------------------------------------------------

    remapped_state_dict = {}


    for key, value in state_dict.items():

        if key.startswith(
            "decoder.decoder."
        ):

            new_key = key.replace(
                "decoder.decoder.",
                "decoder.net.",
                1
            )


            remapped_state_dict[
                new_key
            ] = value

        else:

            remapped_state_dict[
                key
            ] = value


    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        remapped_state_dict
    )


    model.to(
        device
    )


    model.eval()


    return model