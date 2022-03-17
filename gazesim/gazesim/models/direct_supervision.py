import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from gazesim.models.layers import LoadableModule


# from https://arxiv.org/abs/1705.02544

class DirectSupervisionAttention(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # get VGG layers (need all except last pooling operation)
        vgg16 = models.vgg16(True)  # TODO: SHOULD BE SET TO TRUE (not that it's likely to matter)
        modules = [layer for layer in vgg16.features[:-1]]
        # print(len(modules))

        # divide layers into multiple parts of the encoder
        self.encoder_part_0 = nn.Sequential(*modules[:16])
        self.encoder_part_1 = nn.Sequential(*modules[16:23])
        self.encoder_part_2 = nn.Sequential(*modules[23:])

        # create the decoder layers
        self.decoder_part_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
            # nn.Sigmoid(),
        )
        self.decoder_part_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
            # nn.Sigmoid(),
        )
        self.decoder_part_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
            # nn.Sigmoid(),
        )

        # define the single fusion convolution
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x_feat_0 = self.encoder_part_0(x["input_image_0"])
        x_pred_0 = self.decoder_part_0(x_feat_0)

        x_feat_1 = self.encoder_part_1(x_feat_0)
        x_pred_1 = self.decoder_part_1(x_feat_1)

        x_feat_2 = self.encoder_part_2(x_feat_1)
        x_pred_2 = self.decoder_part_2(x_feat_2)

        x_final = torch.cat((x_pred_0, x_pred_1, x_pred_2), dim=1)
        x_final = self.fusion_layer(x_final)

        out = {
            "output_attention": {
                "scale_large": x_pred_0,
                "scale_medium": x_pred_1,
                "scale_small": x_pred_2,
                "final": x_final,
            }
        }
        return out


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    sample = {
        "input_image_0": torch.zeros((1, 3, 224, 224)).to(device),
    }

    net = DirectSupervisionAttention()
    net.to(device)
    result = net(sample)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters for direct supervision network:", num_params)

    # print(net)
