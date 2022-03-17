import torch
import torch.nn as nn

from gazesim.models.layers import LoadableModule

# all of this basically taken from https://github.com/ndrplz/dreyeve/blob/master/experiments/train/models.py
# TODO: maybe consider replacing the C3D parts?
#  see here for alternatives: https://discuss.pytorch.org/t/pertained-c3d-model-for-video-classification/15506


class CoarseSaliencyModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.conv_pool_0 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        # TODO: not sure how to replicate Keras' "same" padding mode in all cases
        #  => "valid" should just be the default that is done in PyTorch anyway

        # TODO: also not entirely sure about which dimensions the tuples refer to (but I'm guessing the first is
        #  the temporal dimension) => mostly matters for the max-pooling that uses different values

        self.conv_pool_1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.conv_pool_2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.conv_pool_3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        )

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x):
        x = self.conv_pool_0(x)
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.conv_pool_3(x)
        x = x.reshape(tuple(x.shape[:2]) + tuple(x.shape[3:]))
        x = self.upsampling(x)
        return x


class SaliencyBranch(LoadableModule):

    def __init__(self, config):
        super().__init__()

        self.coarse_predictor = CoarseSaliencyModel(config)

        self.upsampling_coarse_to_fine = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.conv_refine_0 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.001)
        )

        self.conv_refine_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.001)
        )

        self.conv_refine_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.001)
        )

        self.conv_refine_3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.conv_output_crop = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: need some way to "tell" the current saliency branch that it should work with e.g. input_image_0,
        #  not sure this is the best way but it should work without adding anything to the config or the like
        if "input_image_0" in x:
            x = x["input_image_0"]

        full_x = self.coarse_predictor(x["stack"])
        # print(full_x.shape)
        full_x = self.upsampling_coarse_to_fine(full_x)
        # print(full_x.shape)
        full_x = torch.cat([full_x, x["last_frame"]], dim=1)
        # print(full_x.shape)
        full_x = self.conv_refine_0(full_x)
        # print(full_x.shape)
        full_x = self.conv_refine_1(full_x)
        # print(full_x.shape)
        full_x = self.conv_refine_2(full_x)
        # print(full_x.shape)
        full_x = self.conv_refine_3(full_x)
        # print(full_x.shape)
        # print()

        crop_x = self.coarse_predictor(x["stack_crop"])
        # print(crop_x.shape)
        crop_x = self.conv_output_crop(crop_x)
        # print(crop_x.shape)

        out = {
            "output_attention": full_x,
            "output_attention_crop": crop_x
        }
        return out


class DrEYEveNet(LoadableModule):

    def __init__(self, config):
        super().__init__()

        self.image_branch = SaliencyBranch(config)
        self.optical_flow_branch = SaliencyBranch(config)

        self.final_activation = nn.ReLU()

    def forward(self, x):
        image_x = self.image_branch(x)
        optical_flow_x = self.optical_flow_branch(x)
        combined_x = image_x + optical_flow_x

        predictions = self.final_activation(combined_x)

        out = {"output_attention": predictions}
        return out

    def load_model_info(self, model_info_dict):
        raise NotImplementedError


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    """
    image = torch.ones((2, 3, 150, 200)).to(device)
    state = torch.ones((2, 9)).to(device)
    label = torch.LongTensor([0, 1]).to(device)
    X = {
        "input_image_0": image,
        "input_image_1": image.clone(),
        "input_state": state,
        "label_high_level": label
    }
    """
    X = {
        "stack": torch.zeros((1, 3, 16, 112, 112)).to(device),
        "stack_crop": torch.zeros((1, 3, 16, 112, 112)).to(device),
        "last_frame": torch.zeros((1, 3, 448, 448)).to(device)
    }

    net = SaliencyBranch({}).to(device)
    result = net(X)
    # print(result)

