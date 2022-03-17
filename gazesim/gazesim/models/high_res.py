import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from gazesim.models.layers import LoadableModule, DummyLayer


# from https://pdfs.semanticscholar.org/04ce/787f6cf139c72d5b9cbb47be26f122f19c1b.pdf

class HighResAttention(LoadableModule):

    def __init__(self, config=None):
        super().__init__()

        # defining some stuff for
        self.full_res_size = np.array(config["resize"] if isinstance(config["resize"], tuple)
                                      else (config["resize"], int((config["resize"] / 600.0) * 800.0)), dtype=np.int32)
        self.half_res_size = (np.floor((self.full_res_size - 3 + 2) / 2) + 1).astype(np.int32)
        self.quarter_res_size = (np.floor((self.half_res_size - 3 + 2) / 2) + 1).astype(np.int32)

        # scale factor for increasing the depth of the feature maps
        self.csf = config["channel_scale_factor"]

        # activation layer at the end
        self.final_activation = config["high_res_activation"]

        ##########################
        # FULL RESOLUTION BLOCKS #
        ##########################
        self.full_res_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=4 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.full_res_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.full_res_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=24 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.full_res_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=56 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        #######################################
        # FULL RESOLUTION DOWNSAMPLING LAYERS #
        #######################################
        self.full_to_half_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.full_to_half_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.full_to_half_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.full_to_half_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )

        self.full_to_quarter_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(4, 4), padding=(1, 1)),
            nn.ReLU(),
        )
        self.full_to_quarter_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(4, 4), padding=(1, 1)),
            nn.ReLU(),
        )
        self.full_to_quarter_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(4, 4), padding=(1, 1)),
            nn.ReLU(),
        )

        self.full_to_eighth_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(8, 8), padding=(1, 1)),
            nn.ReLU(),
        )
        self.full_to_eighth_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(8, 8), padding=(1, 1)),
            nn.ReLU(),
        )

        ###############################
        # FULL RESOLUTION FINAL LAYER #
        ###############################
        self.full_final = nn.Sequential(
            nn.Conv2d(in_channels=120 * self.csf, out_channels=8 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        ##########################
        # HALF RESOLUTION BLOCKS #
        ##########################
        self.half_res_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=8 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.half_res_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=24 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.half_res_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=56 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        #######################################
        # HALF RESOLUTION DOWNSAMPLING LAYERS #
        #######################################
        self.half_to_quarter_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.half_to_quarter_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.half_to_quarter_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )

        self.half_to_eighth_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(4, 4), padding=(1, 1)),
            nn.ReLU(),
        )
        self.half_to_eighth_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=16 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(4, 4), padding=(1, 1)),
            nn.ReLU(),
        )

        ####################################
        # HALF RESOLUTION UPSAMPLING LAYER #
        ####################################
        self.half_to_full = nn.UpsamplingBilinear2d(size=tuple(self.full_res_size))

        ###############################
        # HALF RESOLUTION FINAL LAYER #
        ###############################
        self.half_final = nn.Sequential(
            nn.Conv2d(in_channels=120 * self.csf, out_channels=16 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        #############################
        # QUARTER RESOLUTION BLOCKS #
        #############################
        self.quarter_res_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=24 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.quarter_res_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=56 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        ##########################################
        # QUARTER RESOLUTION DOWNSAMPLING LAYERS #
        ##########################################
        self.quarter_to_eighth_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.quarter_to_eighth_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=32 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )

        ########################################
        # QUARTER RESOLUTION UPSAMPLING LAYERS #
        ########################################
        self.quarter_to_full = nn.UpsamplingBilinear2d(size=tuple(self.full_res_size))
        self.quarter_to_half = nn.UpsamplingBilinear2d(size=tuple(self.half_res_size))

        ##################################
        # QUARTER RESOLUTION FINAL LAYER #
        ##################################
        self.quarter_final = nn.Sequential(
            nn.Conv2d(in_channels=120 * self.csf, out_channels=32 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        ###########################
        # EIGHTH RESOLUTION BLOCK #
        ###########################
        self.eighth_res_block_0 = nn.Sequential(
            nn.Conv2d(in_channels=56 * self.csf, out_channels=64 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * self.csf, out_channels=64 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * self.csf, out_channels=64 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * self.csf, out_channels=64 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 * self.csf, out_channels=64 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        #######################################
        # EIGHTH RESOLUTION UPSAMPLING LAYERS #
        #######################################
        self.eighth_to_full = nn.UpsamplingBilinear2d(size=tuple(self.full_res_size))
        self.eighth_to_half = nn.UpsamplingBilinear2d(size=tuple(self.half_res_size))
        self.eighth_to_quarter = nn.UpsamplingBilinear2d(size=tuple(self.quarter_res_size))

        #################################
        # EIGHTH RESOLUTION FINAL LAYER #
        #################################
        self.eighth_final = nn.Sequential(
            nn.Conv2d(in_channels=120 * self.csf, out_channels=64 * self.csf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        ########################
        # COMBINED FINAL LAYER #
        ########################
        self.combined_final = nn.Sequential(
            nn.Conv2d(in_channels=120 * self.csf, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Hardtanh() if self.final_activation else DummyLayer(),
        )

    def forward(self, x):
        # this is structured such that the most high-res stuff is done first

        # first and second full res blocks
        x_full_0 = self.full_res_block_0(x["input_image_0"])
        x_full_1 = self.full_res_block_1(x_full_0)

        # first half res block
        x_full_to_half_0 = self.full_to_half_block_0(x_full_0)
        x_half_0 = self.half_res_block_0(x_full_to_half_0)

        # third full res block
        x_half_to_full_0 = self.half_to_full(x_half_0)
        x_full_2 = self.full_res_block_2(torch.cat((x_full_1, x_half_to_full_0), dim=1))

        # second half res block
        x_full_to_half_1 = self.full_to_half_block_1(x_full_1)
        x_half_1 = self.half_res_block_1(torch.cat((x_full_to_half_1, x_half_0), dim=1))

        # first quarter res block
        x_full_to_quarter_0 = self.full_to_quarter_block_0(x_full_1)
        x_half_to_quarter_0 = self.half_to_quarter_block_0(x_half_0)
        x_quarter_0 = self.quarter_res_block_0(torch.cat((x_full_to_quarter_0, x_half_to_quarter_0), dim=1))

        # fourth full res block
        x_half_to_full_1 = self.half_to_full(x_half_1)
        x_quarter_to_full_0 = self.quarter_to_full(x_quarter_0)
        x_full_3 = self.full_res_block_3(torch.cat((x_full_2, x_half_to_full_1, x_quarter_to_full_0), dim=1))

        # third half res block
        x_full_to_half_2 = self.full_to_half_block_2(x_full_2)
        x_quarter_to_half_0 = self.quarter_to_half(x_quarter_0)
        x_half_2 = self.half_res_block_2(torch.cat((x_full_to_half_2, x_half_1, x_quarter_to_half_0), dim=1))

        # second quarter res block
        x_full_to_quarter_1 = self.full_to_quarter_block_1(x_full_2)
        x_half_to_quarter_1 = self.half_to_quarter_block_1(x_half_1)
        x_quarter_1 = self.quarter_res_block_1(torch.cat((x_full_to_quarter_1, x_half_to_quarter_1, x_quarter_0), dim=1))

        # first (and only) eighth res block
        x_full_to_eighth_0 = self.full_to_eighth_block_0(x_full_2)
        x_half_to_eighth_0 = self.half_to_eighth_block_0(x_half_1)
        x_quarter_to_eighth_0 = self.quarter_to_eighth_block_0(x_quarter_0)
        x_eighth_0 = self.eighth_res_block_0(torch.cat((x_full_to_eighth_0, x_half_to_eighth_0, x_quarter_to_eighth_0), dim=1))

        # final full res layer
        x_half_to_full_2 = self.half_to_full(x_half_2)
        x_quarter_to_full_1 = self.quarter_to_full(x_quarter_1)
        x_eighth_to_full_0 = self.eighth_to_full(x_eighth_0)
        x_full_final = self.full_final(torch.cat((x_full_3, x_half_to_full_2, x_quarter_to_full_1, x_eighth_to_full_0), dim=1))

        # final half res layer + upsampling
        x_full_to_half_3 = self.full_to_half_block_3(x_full_3)
        x_quarter_to_half_1 = self.quarter_to_half(x_quarter_1)
        x_eighth_to_half_0 = self.eighth_to_half(x_eighth_0)
        x_half_final = self.half_final(torch.cat((x_full_to_half_3, x_half_2, x_quarter_to_half_1, x_eighth_to_half_0), dim=1))
        x_half_final = self.half_to_full(x_half_final)

        # final quarter res layer + upsampling
        x_full_to_quarter_2 = self.full_to_quarter_block_2(x_full_3)
        x_half_to_quarter_2 = self.half_to_quarter_block_2(x_half_2)
        x_eighth_to_quarter_0 = self.eighth_to_quarter(x_eighth_0)
        x_quarter_final = self.quarter_final(torch.cat((x_full_to_quarter_2, x_half_to_quarter_2, x_quarter_1, x_eighth_to_quarter_0), dim=1))
        x_quarter_final = self.quarter_to_full(x_quarter_final)

        # final eighth res layer + upsampling
        x_full_to_eighth_1 = self.full_to_eighth_block_1(x_full_3)
        x_half_to_eighth_1 = self.half_to_eighth_block_1(x_half_2)
        x_quarter_to_eighth_1 = self.quarter_to_eighth_block_1(x_quarter_1)
        x_eighth_final = self.eighth_final(torch.cat((x_full_to_eighth_1, x_half_to_eighth_1, x_quarter_to_eighth_1, x_eighth_0), dim=1))
        x_eighth_final = self.eighth_to_full(x_eighth_final)

        # final concatenation and stuff
        x_combined_final = torch.cat((x_full_final, x_half_final, x_quarter_final, x_eighth_final), dim=1)
        x_combined_final = self.combined_final(x_combined_final)

        out = {
            "output_attention": x_combined_final
        }
        return out


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    sample = {
        "input_image_0": torch.zeros((1, 3, 300, 400)).to(device),
    }

    net = HighResAttention({"resize": 300, "channel_scale_factor": 1}).to(device)
    result = net(sample)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters for high-res network:", num_params)
