import torch
import torch.nn.functional as F


def _image_apply_function(tensor, function, **kwargs):
    # assumes that the last two dimensions are the image dimensions H x W
    original_shape = tuple(tensor.shape)
    tensor = torch.reshape(tensor, original_shape[:-2] + (-1,))
    tensor = function(tensor, **kwargs)
    tensor = torch.reshape(tensor, original_shape)
    return tensor


def _image_apply_reduction(tensor, reduction, **kwargs):
    # assumes that the last two dimensions are the image dimensions H x W
    original_shape = tuple(tensor.shape)
    tensor = torch.reshape(tensor, original_shape[:-2] + (-1,))
    tensor = reduction(tensor, **kwargs)
    return tensor


def image_max(tensor):
    tensor, _ = _image_apply_reduction(tensor, torch.max, dim=-1)
    return tensor


def image_softmax(tensor):
    return _image_apply_function(tensor, F.softmax, dim=-1)


def image_log_softmax(tensor):
    return _image_apply_function(tensor, F.log_softmax, dim=-1)


def convert_attention_to_image(attention, out_shape=None):
    assert len(attention.shape) == 4, \
        "Tensor must have exactly 4 dimensions (batch, channel, height, width in some order)."

    # make sure that we are not doing integer types (because of the division)
    attention = attention.double()

    # make sure all the dimensions are correct
    if not attention.shape[1] == 3 and attention.shape[3] == 3:
        attention = attention.permute(0, 3, 1, 2)

    # resize the image if necessary
    if out_shape is not None and (attention.shape[2] != out_shape[0] or attention.shape[3] != out_shape[1]):
        attention = torch.nn.functional.interpolate(attention, out_shape, mode="bicubic", align_corners=True)

    # TODO: maybe also add color channels

    # divide by the maximum
    maximum = image_max(attention).unsqueeze(-1).unsqueeze(-1)
    return attention / maximum

"""
def load_model(checkpoint_path, gpu=-1, return_config=False):
    # define paths
    log_dir = os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), os.pardir))
    config_path = os.path.join(log_dir, "config.json")

    # load the config (only needed to get info, probably not returned with the model?)
    with open(config_path, "r") as f:
        train_config = json.load(f)
    train_config["gpu"] = gpu
    train_config["model_load_path"] = checkpoint_path
    train_config = parse_config(train_config)

    # load the model
    model_info = train_config["model_info"]
    model = resolve_model_class(train_config["model_name"])(train_config)
    model.load_state_dict(model_info["model_state_dict"])
    # model.load_model_info(model_info)
    # model = model.to(device)  # I guess device should probably be identified outside of the thingy?

    if return_config:
        return model, train_config
    return model
"""
