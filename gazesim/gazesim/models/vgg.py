import torch
import torch.nn as nn
import torchvision.models as models

# from torchsummary import summary


class VGG16BaseModel(nn.Module):

    def __init__(self, transfer_depth=16, transfer_weights=True):
        super(VGG16BaseModel, self).__init__()
        # TODO: proper inputs instead of hard-coding channel sizes etc.

        # defining the feature-extracting CNN using VGG16 layers as a basis
        vgg16 = models.vgg16(transfer_weights)
        modules = []
        for layer in vgg16.features[:transfer_depth]:
            modules.append(layer)

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 512, 18, 25] after this with transfer_depth 17-23
        # shape will be [-1, 256, 37, 50] after this with transfer_depth 16

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            # nn.Upsample(size=(36, 50)),
            # nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.Softmax2d()
            # no ReLU because the outputs are supposed to be log-probabilities for KL-divergence
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upscaling(x)
        return x


if __name__ == "__main__":
    tensor = torch.zeros((1, 3, 150, 200))

    net = VGG16BaseModel()
    out = net(tensor)

    # summary(net, (3, 150, 200))
