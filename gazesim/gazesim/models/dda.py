import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from gazesim.models.layers import ControlActivationLayer, LoadableModule, DummyLayer, Conv1dSamePadding


class DDAModel(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        self.features_point_net = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.InstanceNorm2d(num_features=32, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.InstanceNorm2d(num_features=64, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.InstanceNorm2d(num_features=128, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.AdaptiveAvgPool2d((8, 1)),
        )

        self.features_merge_net = nn.Sequential(
            Conv1dSamePadding(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            Conv1dSamePadding(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            Conv1dSamePadding(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            Conv1dSamePadding(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),  # TODO: should this really have that many in_features?
        )

        # TODO: potentially make this configurable depending on the number of input state variables
        self.states_conv_net = nn.Sequential(
            Conv1dSamePadding(in_channels=30, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            Conv1dSamePadding(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            Conv1dSamePadding(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=1e-2),
            Conv1dSamePadding(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=1, dilation=1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
        )

        self.control_regressor = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.Linear(in_features=32, out_features=4),  # TODO: activation?
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

    def forward(self, x):
        # feature tracks passed "separately" through PointNet, then merged with 1D convolution network
        feature_track_x = self.features_point_net(x["input_feature_tracks"]["stack"])
        feature_track_x = feature_track_x.reshape(feature_track_x.shape[:-1])
        feature_track_x = self.features_merge_net(feature_track_x)

        # state reference and estimate passed "jointly" through one network (no interaction between "channels" though)
        state_x = self.states_conv_net(x["input_state"]["stack"])

        # final control command regression
        control_x = torch.cat((feature_track_x, state_x), dim=-1)
        control_x = self.control_regressor(control_x)
        out = {"output_control": control_x}

        """
        print("PointNet")
        feature_track_x = x["input_feature_tracks"]["stack"]
        for layer_idx, layer in enumerate(self.test_point_net):
            print(f"Layer {layer_idx}, before: {feature_track_x.shape}")
            feature_track_x = layer(feature_track_x)
            print(f"Layer {layer_idx}, after: {feature_track_x.shape}\n")
        feature_track_x = feature_track_x.reshape(feature_track_x.shape[:-1])

        print("MergeNet")
        for layer_idx, layer in enumerate(self.test_merge_net):
            print(f"Layer {layer_idx}, before: {feature_track_x.shape}")
            feature_track_x = layer(feature_track_x)
            print(f"Layer {layer_idx}, after: {feature_track_x.shape}\n")
            
        state_x = x["input_state"]["stack"]
        for layer_idx, layer in enumerate(self.test_states_conv_net):
            print(f"Layer {layer_idx}, before: {state_x.shape}")
            state_x = layer(state_x)
            print(f"Layer {layer_idx}, after: {state_x.shape}\n")
        """

        return out


if __name__ == "__main__":
    test_input = {
        "input_feature_tracks": {
            "stack": torch.zeros((1, 5, 8, 40)),
        },
        "input_state": {
            "stack": torch.zeros((1, 30, 8)),
        },
    }

    net = DDAModel()
    test_output = net(test_input)
    print(test_output)
