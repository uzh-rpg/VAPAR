import torch
import torch.nn as nn
import torchvision.models as models

from gazesim.models.layers import ControlActivationLayer, LoadableModule, DummyLayer


class ResNetLargerGRUCell(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet = models.resnet18(True)
        modules = list(resnet.children())[:7]

        self.features = nn.Sequential(*modules)
        self.downsample = nn.Conv2d(256, 256, 3, stride=2)

        # linear layers after flattening
        self.image_fc_0 = nn.Linear(6144, 512)
        self.image_fc_1 = nn.Linear(512, 256)

        # GRU cell
        self.gru = nn.GRUCell(256, 256)

        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.5)

    def forward(self, x, h):
        image_x = self.features(x)
        image_x = self.relu(self.downsample(image_x))
        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.relu(self.fc_dropout(self.image_fc_0(image_x)))
        image_x = self.relu(self.fc_dropout(self.image_fc_1(image_x)))

        gru_x = self.gru(image_x, h)
        return gru_x


class ResNetLargerGRURegressor(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # RNN cell
        self.rnn_cell = ResNetLargerGRUCell(config)

        #
        self.hidden_size = 256

        # the regressor after the RNN
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 4),
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

    def get_init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x, h=None, return_hidden=False):
        image_x = x["input_image_0"]["stack"]
        image_x = image_x.permute(0, 2, 1, 3, 4)
        sequence_length = image_x.size(1)  # need to check this again

        hidden_x = self.get_init_state(image_x.size(0)) if h is None else h
        hidden_x = hidden_x.to(image_x.device)
        for i in range(sequence_length):
            hidden_x = self.rnn_cell(image_x[:, i], hidden_x)

        control_inputs = self.regressor(hidden_x)

        out = {"output_control": control_inputs}
        if return_hidden:
            out["output_hidden"] = hidden_x
        return out
