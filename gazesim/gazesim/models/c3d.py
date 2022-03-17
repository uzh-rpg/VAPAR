import torch
import torch.nn as nn

# from torchsummary import summary
from gazesim.models.layers import ControlActivationLayer, LoadableModule, DummyLayer


class C3DRegressor(LoadableModule):
    """
    Based on the implementation in this repository: https://github.com/hermanprawiro/c3d-extractor-pytorch
    (although identical ones can be found in other GitHub repositories; this seems to be the earliest I could find).
    """

    def __init__(self, config=None):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 4)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.final_activation = DummyLayer() if config["no_control_activation"] else ControlActivationLayer()

    def forward(self, x):

        h = self.relu(self.conv1(x["input_image_0"]["stack"]))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.fc6(h)
        h = self.relu(h)

        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probabilities = self.final_activation(logits)

        out = {"output_control": probabilities}
        return out


class C3DStateRegressor(C3DRegressor):
    """
    Based on the implementation in this repository: https://github.com/hermanprawiro/c3d-extractor-pytorch
    (although identical ones can be found in other GitHub repositories; this seems to be the earliest I could find).
    """

    def __init__(self, config=None):
        super().__init__()

        self.conv_fc_0 = nn.Linear(8192, 1024)
        self.conv_fc_1 = nn.Linear(1024, 512)

        self.state_fc_0 = nn.Linear(9, 256)
        self.state_fc_1 = nn.Linear(256, 256)

        self.combined_fc_0 = nn.Linear(512 + 256, 4)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.final_activation = DummyLayer() if config["no_control_activation"] else ControlActivationLayer()

    def forward(self, x):
        image_x = self.relu(self.conv1(x["input_image_0"]["stack"]))
        image_x = self.pool1(image_x)

        image_x = self.relu(self.conv2(image_x))
        image_x = self.pool2(image_x)

        image_x = self.relu(self.conv3a(image_x))
        image_x = self.relu(self.conv3b(image_x))
        image_x = self.pool3(image_x)

        image_x = self.relu(self.conv4a(image_x))
        image_x = self.relu(self.conv4b(image_x))
        image_x = self.pool4(image_x)

        image_x = self.relu(self.conv5a(image_x))
        image_x = self.relu(self.conv5b(image_x))
        image_x = self.pool5(image_x)

        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.relu(self.dropout(self.conv_fc_0(image_x)))
        image_x = self.relu(self.dropout(self.conv_fc_1(image_x)))

        state_x = self.relu(self.dropout(self.state_fc_0(x["input_state"])))
        state_x = self.relu(self.dropout(self.state_fc_1(state_x)))

        combined_x = torch.cat([image_x, state_x], dim=-1)

        logits = self.combined_fc_0(combined_x)
        probabilities = self.final_activation(logits)

        out = {"output_control": probabilities}
        return out


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    tensor = torch.zeros((1, 3, 16, 112, 112)).to(device)

    X = {
        "input_image_0": {"stack": torch.zeros((1, 3, 16, 112, 112)).to(device)},
        "input_state": torch.zeros((1, 9)).to(device)
    }

    net = C3DStateRegressor().to(device)
    out = net(X)
    print(out["output_control"].shape)

    # summary(net, (3, 16, 300, 400), batch_size=1)
