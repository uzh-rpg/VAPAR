import numpy as np
import torch
import torchvision.transforms.functional as tvf

from torchvision import transforms


class MultiRandomApply(transforms.RandomApply):

    def forward(self, img):
        for t in self.transforms:
            if torch.rand(1) < self.p:
                img = t(img)
        return img


class ImageToAttentionMap:

    def __call__(self, sample):
        # for now always expects numpy array (and respective dimension for the colour channels)
        return sample[:, :, 0]


class MakeValidDistribution:

    def __call__(self, sample):
        # expects torch tensor with
        pixel_sum = torch.sum(sample, [])
        if pixel_sum > 0.0:
            sample = sample / pixel_sum
        return sample


class ManualRandomCrop:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.template_image = torch.zeros((3,) + tuple(self.input_size))
        self.current_parameters = transforms.RandomCrop.get_params(self.template_image, output_size=self.output_size)

    def __call__(self, img):
        return tvf.crop(img, *self.current_parameters)

    def update(self):
        self.current_parameters = transforms.RandomCrop.get_params(self.template_image, output_size=self.output_size)


class GaussianNoise:

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, img):
        # it is assumed that the image has already been converted to range [0.0, 1.0]
        img = img + self.sigma * torch.randn_like(img)
        img = torch.clamp(img, 0.0, 1.0)
        return img


class DrEYEveTransform(object):

    def __init__(
            self,
            input_names,
            full_size=(448, 448),
            small_size=(112, 112),
            pre_crop_size=(256, 256)
    ):
        self.input_names = input_names
        self.full_size = full_size
        self.small_size = small_size
        self.pre_crop_size = pre_crop_size

        self.random_crop = ManualRandomCrop(pre_crop_size, small_size)

        self.input_transforms = {
            "stack": [transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.small_size),
                transforms.ToTensor(),
            ]) for _ in input_names],
            "stack_crop": [transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.full_size),
                transforms.ToTensor(),
            ]) for _ in input_names],
            "last_frame": [transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.full_size),
                transforms.ToTensor(),
            ]) for _ in input_names]
        }
        self.output_transforms = {
            "stack": transforms.Compose([
                ImageToAttentionMap(),
                transforms.ToPILImage(),
                transforms.Resize(self.full_size),
                transforms.ToTensor(),
                MakeValidDistribution()
            ]),
            "stack_crop": transforms.Compose([
                ImageToAttentionMap(),
                transforms.ToPILImage(),
                transforms.Resize(self.small_size),
                self.random_crop,
                transforms.ToTensor(),
                MakeValidDistribution()  # not sure if this will work with the cropping stuff
            ])
        }

    def _update(self):
        self.random_crop.update()

    def _apply_input_transforms(self, sample):
        # sample should already be a dictionary containing the image stack (without the stacking part?)
        # => guess it's fine if it's any iterable, including tensor?
        for idx in range(len(self.input_names)):
            current_key = f"input_image_{idx}"
            current = {"stack": [], "stack_crop": [], "last_frame": None}
            stack_size = len(sample[current_key])
            for img_idx, img in enumerate(sample[current_key]):
                if img_idx == stack_size - 1:
                    current["last_frame"] = self.input_transforms["last_frame"][idx](img)
                current["stack"].append(self.input_transforms["stack"][idx](img))
                current["stack_crop"].append(self.input_transforms["stack_crop"][idx](img))

            for k in current:
                if k != "last_frame":
                    current[k] = torch.stack(current[k], 1)
            sample[current_key] = current
        return sample

    def _apply_output_transforms(self, sample):
        sample["output_attention"] = {k: self.output_transforms[k](sample["output_attention"])
                                      for k in self.output_transforms}
        return sample

    def apply_transforms(self, sample):
        self._update()
        sample = self._apply_input_transforms(sample)
        sample = self._apply_output_transforms(sample)
        return sample

    def __call__(self, sample):
        # what do we expect from sample? (that it is a dictionary, obviously...)
        # - input_image_X (multiple might exist, but they should all be treated the same)
        # - output_attention
        # - do we expect that everything is stacked already? might be a good thing, yeah
        # - think that's pretty much it...

        # the same random crop should be applied to all input images and the output image

        # first determine the random crop parameters
        # TODO: not sure if this will work, maybe the colour dimension needs to be somewhere else?
        dummy = tvf.to_pil_image(np.zeros(self.pre_crop_size + (3,)))
        i, j, h, w = transforms.RandomCrop.get_params(dummy, output_size=self.small_size)
        # TODO: a different solution could be to write a custom RandomCrop class that has a method that needs to be
        #  called to update the current crop parameters, which could be done before transforming every sample...

        # apply transforms to input images and stack them together
        for idx, _ in enumerate(self.input_names):
            current_key = f"input_image_{idx}"
            current = {"stack": [], "stack_crop": [], "last_frame": None}
            stack_size = sample[current_key]
            for img_idx, img in enumerate(sample[current_key]):
                if img_idx == stack_size - 1:
                    current["last_frame"] = self.last_frame_transforms[idx](img)
                else:
                    current["stack"].append(self.stack_transforms[idx](img))
                    current["stack_crop"].append(tvf.crop(self.stack_crop_transforms[idx](img), i, j, h, w))

            for k in current:
                if k != "last_frame":
                    current[k] = torch.stack([k], 1)
            sample[current_key] = current

        # finally, apply transforms to output
        sample["output_attention"] = {k: tvf.crop(self.output_transforms[k](sample["output_attention"]), i, j, h, w)
                                      for k in self.output_transforms}

        return sample


if __name__ == "__main__":
    import cv2

    init_transform = transforms.Compose([
        transforms.ToPILImage()
    ])

    test_input = np.zeros((300, 400, 3), dtype="uint8")
    test_input[120:180, 170:230, :] = 255
    test_input = init_transform(test_input)

    transform = ManualRandomCrop((300, 400), (150, 200))

    crop_0 = transform(test_input)
    crop_0.show(title="crop 0")

    crop_1 = transform(test_input)
    crop_1.show(title="crop 1")

    transform.update()
    crop_2 = transform(test_input)
    crop_2.show(title="crop 2")
