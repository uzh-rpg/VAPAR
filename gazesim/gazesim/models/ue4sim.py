import torch
import torch.nn as nn

from gazesim.models.layers import ControlActivationLayer, LoadableModule, DummyLayer


class UE4SimRegressor(LoadableModule):

    # architecture from https://arxiv.org/pdf/1708.05884.pdf

    def __init__(self, config=None):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=28, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features=1680, out_features=1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=4),
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

    def forward(self, x):
        image_x = self.feature_extractor(x["input_image_0"])
        image_x = image_x.reshape(image_x.size(0), -1)

        controls = self.regressor(image_x)

        out = {"output_control": controls}
        return out


if __name__ == "__main__":
    network = UE4SimRegressor({"no_control_activation": True})
    dummy_input = {
        "input_image_0": torch.zeros((1, 3, 180, 320)),
    }

    dummy_output = network(dummy_input)
