import torch
import torch.nn as nn

from gazesim.models.utils import image_softmax


class LoadableModule(nn.Module):

    def load_model_info(self, model_info_dict):
        self.load_state_dict(model_info_dict["model_state_dict"])


class ControlActivationLayer(nn.Module):

    def forward(self, x):
        # split into thrust and other commands and perform activation
        thrust_tensor = torch.sigmoid(x[:, 0:1])
        other_tensor = torch.tanh(x[:, 1:])

        # recombine and return result
        x = torch.cat((thrust_tensor, other_tensor), dim=1)
        return x


class DummyLayer(nn.Module):

    def forward(self, x):
        return x


class Conv1dSamePadding(nn.Conv1d):

    def forward(self, x):
        x_shape_last_dim = x.shape[-1]
        x = super(Conv1dSamePadding, self).forward(x)
        x = x[..., :x_shape_last_dim]
        return x


class ImageBCELoss(torch.nn.BCEWithLogitsLoss):

    def forward(self, prediction, target):
        prediction = torch.reshape(prediction, (prediction.shape[0], -1))
        target = torch.reshape(target, (target.shape[0], -1))
        loss = super(ImageBCELoss, self).forward(prediction, target)
        return loss
