import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from torchsummary import summary
from gazesim.models.layers import ControlActivationLayer, LoadableModule, DummyLayer


class ResNet18BaseModel(LoadableModule):

    def __init__(self, module_transfer_depth=7, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300
        # shape will be [-1, 256, 10, 13] after this with module_transfer_depth 7 and input height 150

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            nn.Upsample(size=(37, 50)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upscaling(x)
        return x


class ResNet18BaseModelSimple(LoadableModule):

    def __init__(self, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 128, 38, 50] after this with module_transfer_depth 6 and input height 300

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            # nn.Upsample(size=(37, 50)),
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upscaling(x)
        return x


class ResNet18Regressor(LoadableModule):

    def __init__(self, module_transfer_depth=7, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(256, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.not_sure(x)
        # print(x.shape)
        x = self.pooling(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x


class ResNet18SimpleRegressor(LoadableModule):

    def __init__(self, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 128, 38, 50] after this with module_transfer_depth 6 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(128, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x


class ResNet18DualBranchRegressor(LoadableModule):

    def __init__(self, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        # get the layers form ResNet
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        # creating the feature extractors/two branches
        self.features_0 = nn.Sequential(*modules)
        self.features_1 = nn.Sequential(*modules)

        # pooling layer, using the same as ResNet for now
        # should be able to apply this to features from both branches
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # regressor with double the input size as simple ResNet regressor
        self.regressor = nn.Sequential(
            nn.Linear(256, 4),
            ControlActivationLayer()
        )

    def forward(self, x):
        # pass both images through the branches
        x_0 = self.features_0(x[0])
        x_1 = self.features_1(x[1])

        # pooling
        x_0 = self.pooling(x_0)
        x_1 = self.pooling(x_1)

        # flatten the pooled features
        x_0 = x_0.reshape(x_0.size(0), -1)
        x_1 = x_1.reshape(x_1.size(0), -1)

        # concatenate the features and pass them through the regressor
        x = torch.cat((x_0, x_1), 1)
        x = self.regressor(x)

        return x


class ResNetStateRegressor(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # state "transformation" layer
        self.state = nn.Linear(len(config["drone_state_names"]), 256)

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(512, 4),
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.pooling(image_x)
        image_x = image_x.reshape(image_x.size(0), -1)

        state_x = self.state(x["input_state"])

        combined_x = torch.cat([image_x, state_x], dim=-1)

        probabilities = self.regressor(combined_x)

        out = {"output_control": probabilities}
        return out


class ResNetStateLargerRegressor(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features = nn.Sequential(*modules)
        self.downsample = nn.Conv2d(256, 256, 3, stride=2)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc_0 = nn.Linear(6144, 512)
        self.image_fc_1 = nn.Linear(512, 256)

        # state "transformation" layer
        self.state = nn.Linear(len(config["drone_state_names"]), 256)
        self.state_fc_0 = nn.Linear(256, 256)

        # defining the upscaling layers to get out the original image size again
        # TODO: maybe add branching
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 4),
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.relu(self.downsample(image_x))
        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.relu(self.fc_dropout(self.image_fc_0(image_x)))
        image_x = self.relu(self.fc_dropout(self.image_fc_1(image_x)))

        state_x = self.relu(self.fc_dropout(self.state(x["input_state"])))
        state_x = self.relu(self.fc_dropout(self.state_fc_0(state_x)))

        combined_x = torch.cat([image_x, state_x], dim=-1)

        probabilities = self.regressor(combined_x)

        out = {"output_control": probabilities}
        return out


class ResNetRegressor(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        self.regress_gaze = "gaze" in config["model_name"]

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # defining the upscaling layers to get out the original image size again
        activation = DummyLayer()
        if not self.regress_gaze and not config["no_control_activation"]:
            activation = ControlActivationLayer()
        elif self.regress_gaze and config["gaze_activation"]:
            activation = nn.Tanh()
        self.regressor = nn.Sequential(
            nn.Linear(256, 2 if self.regress_gaze else 4),
            activation,
        )

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.pooling(image_x)
        image_x = image_x.reshape(image_x.size(0), -1)

        probabilities = self.regressor(image_x)

        out = {"output_gaze" if self.regress_gaze else "output_control": probabilities}
        return out


class ResNetLargerRegressor(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        self.regress_gaze = "gaze" in config["model_name"]

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features = nn.Sequential(*modules)
        self.downsample = nn.Conv2d(256, 256, 3, stride=2)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc_0 = nn.Linear(6144, 512)
        self.image_fc_1 = nn.Linear(512, 256)

        # defining the upscaling layers to get out the original image size again
        activation = DummyLayer()
        if not self.regress_gaze and not config["no_control_activation"]:
            activation = ControlActivationLayer()
        elif self.regress_gaze and config["gaze_activation"]:
            activation = nn.Tanh()
        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 2 if self.regress_gaze else 4),
            activation
        )

        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.relu(self.downsample(image_x))
        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.relu(self.fc_dropout(self.image_fc_0(image_x)))
        image_x = self.relu(self.fc_dropout(self.image_fc_1(image_x)))

        probabilities = self.regressor(image_x)

        out = {"output_gaze" if self.regress_gaze else "output_control": probabilities}
        return out


class ResNetLargerRegressorDualBranch(ResNetLargerRegressor):

    def __init__(self, config=None):
        super().__init__(config)

        self.regress_gaze = "gaze" in config["model_name"]

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.features_second = nn.Sequential(*modules)
        self.downsample_second = nn.Conv2d(256, 256, 3, stride=2)
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # pooling layer, using the same as ResNet for now
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc_0 = nn.Linear(6144 * 2, 512)

    def forward(self, x):
        image_x_0 = self.features(x["input_image_0"])
        image_x_0 = self.relu(self.downsample(image_x_0))
        image_x_0 = image_x_0.reshape(image_x_0.size(0), -1)

        image_x_1 = self.features_second(x["input_image_1"])
        image_x_1 = self.relu(self.downsample_second(image_x_1))
        image_x_1 = image_x_1.reshape(image_x_1.size(0), -1)

        combined_image_x = torch.cat([image_x_0, image_x_1], dim=-1)

        image_x = self.relu(self.fc_dropout(self.image_fc_0(combined_image_x)))
        image_x = self.relu(self.fc_dropout(self.image_fc_1(image_x)))

        probabilities = self.regressor(image_x)

        out = {"output_gaze" if self.regress_gaze else "output_control": probabilities}
        return out


class ResNetLargerRegressorMultiHead(ResNetLargerRegressor):

    def __init__(self, config=None):
        super().__init__(config)

        del self.regressor

        # defining the upscaling layers to get out the original image size again
        activation = DummyLayer()
        if not self.regress_gaze and not config["no_control_activation"]:
            activation = ControlActivationLayer()
        elif self.regress_gaze and config["gaze_activation"]:
            activation = nn.Tanh()

        self.branches = nn.ModuleList()
        for b in range(5):
            branch = nn.Sequential(
                nn.Linear(256, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 2 if self.regress_gaze else 4),
                activation
            )
            self.branches.append(branch)

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.relu(self.downsample(image_x))
        image_x = image_x.reshape(image_x.size(0), -1)
        image_x = self.relu(self.fc_dropout(self.image_fc_0(image_x)))
        image_x = self.relu(self.fc_dropout(self.image_fc_1(image_x)))

        # use different branches depending on label of each sample in batch
        samples = []
        index = torch.arange(end=image_x.size(0)).to(image_x.device)
        for b_idx, branch in enumerate(self.branches):
            # get subset of batch where label matches branch index
            subset = (x["label_high_level"] == b_idx)

            # vector of all indices in the batch, reshaped to match the dimensions of the input vector
            sub_index = index[subset]

            # select the correct samples
            branch_batch = torch.index_select(image_x, 0, sub_index)

            # pass them through the branch head
            branch_batch = branch(branch_batch)

            # put them in the samples list
            for s_idx, sample in zip(sub_index, branch_batch):
                samples.append((int(s_idx), sample.unsqueeze(0)))

        # combine samples again
        samples = [sample[1] for sample in sorted(samples, key=lambda s: s[0])]
        probabilities = torch.cat(samples, 0)

        out = {"output_gaze" if self.regress_gaze else "output_control": probabilities}
        return out


class ResNetLargerAttentionAndControl(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # defining the feature-extracting CNN
        resnet18 = models.resnet18(True)
        modules = list(resnet18.children())[:7]

        self.resnet_features = nn.Sequential(*modules)
        self.resnet_downsample = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # shape will be [-1, 256, 19, 25] after this with module_transfer_depth 7 and input height 300

        # linear layers after flattening
        self.control_net = nn.Sequential(
            nn.Linear(5120, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # defining the regressor
        self.control_regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 4),
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

        # defining the upscaling layers to get out the original image size again
        self.attention_upscaling = nn.Sequential(
            nn.Upsample(size=(37, 50)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        image_x = self.resnet_features(x["input_image_0"])

        control_x = self.resnet_downsample(image_x)
        control_x = control_x.reshape(control_x.size(0), -1)
        control_x = self.control_net(control_x)
        control_x = self.control_regressor(control_x)

        image_x = self.attention_upscaling(image_x)

        out = {
            "output_control": control_x,
            "output_attention": image_x
        }
        return out


class StateOnlyRegressor(LoadableModule):

    def __init__(self, config=None):
        super().__init__()
        # state "transformation" layer
        self.state = nn.Linear(len(config["drone_state_names"]), 256)
        self.state_fc_0 = nn.Linear(256, 512)

        # defining the upscaling layers to get out the original image size again
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 4),
            DummyLayer() if config["no_control_activation"] else ControlActivationLayer(),
        )

        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        state_x = self.relu(self.fc_dropout(self.state(x["input_state"])))
        state_x = self.relu(self.fc_dropout(self.state_fc_0(state_x)))

        probabilities = self.regressor(state_x)

        out = {"output_control": probabilities}
        return out


###########################
# VISUAL ATTENTION MODELS #
###########################


class ResNetAttention(LoadableModule):

    def __init__(self, config=None, module_transfer_depth=6, transfer_weights=True):
        super().__init__()

        self.features_only = False if config is None or "resnet_features_only" not in config else config["resnet_features_only"]

        # defining the feature-extracting CNN using VGG16 layers as a basis
        resnet18 = models.resnet18(transfer_weights)
        modules = list(resnet18.children())[:module_transfer_depth]

        self.features = nn.Sequential(*modules)
        # shape will be [-1, 128, 38, 50] after this with module_transfer_depth 6 and input height 300

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            # nn.Upsample(size=(37, 50)),
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(),
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        if self.features_only:
            out = {
                "output_features": image_x,
            }
            return out

        image_x = self.upscaling(image_x)
        out = {
            "output_attention": image_x,
        }
        return out


class SimpleAttention(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # defining the upscaling layers to get out the original image size again
        self.upscaling = nn.Sequential(
            nn.Upsample(size=(75, 100)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(size=(150, 200)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(size=(300, 400)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        image_x = self.features(x["input_image_0"])
        image_x = self.upscaling(image_x)
        out = {
            "output_attention": image_x
        }
        return out


if __name__ == "__main__":
    cfg = {
        "no_control_activation": False,
        "resnet_features_only": True,
        "model_name": "resnet_gaze",
        "gaze_activation": False,
    }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    tensor = torch.zeros((1, 3, 300, 400)).to(device)
    # tensor = [torch.zeros((1, 3, 300, 400)).to(device), torch.zeros((1, 3, 300, 400)).to(device)]
    sample = {
        "input_image_0": torch.zeros((1, 3, 150, 200)).to(device),
        "input_state": torch.zeros((1, 9)).to(device)
    }

    # net = ResNet18DualBranchRegressor().to(device)
    # net = ResNetLargerAttentionAndControl(cfg).to(device)
    # net = ResNetRegressor(cfg).to(device)
    # net = ResNetAttention(cfg).to(device)
    net = ResNetLargerRegressor(cfg).to(device)
    out = net(sample)
    # print(out["output_features"].shape)
    # process = F.adaptive_avg_pool2d(out["output_features"], (1, 1))
    # process = process.reshape(-1)
    # print(process.shape)
    # print(out)
    print(net)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters for ResNet attention network :", num_params)

    # summary(net, (3, 300, 400))

