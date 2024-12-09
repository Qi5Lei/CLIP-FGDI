from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np
from .functional import to_tensor, augmentations

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, crop_factor=0.8, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.crop_factor = crop_factor
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.crop_factor ** 2, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img

        return img


class RandomOcclusion(object):
    def __init__(self, min_size=0.2, max_size=1):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        if self.max_size == 0:
            return img
        H = img.height
        W = img.width
        S = min(H, W)
        s = np.random.randint(max(1, int(S * self.min_size)), int(S * self.max_size))
        x0 = np.random.randint(W - s)
        y0 = np.random.randint(H - s)
        x1 = x0 + s
        y1 = y0 + s
        block = Image.new('RGB', (s, s), (255, 255, 255))
        img.paste(block, (x0, y0, x1, y1))
        return img


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob           = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width  = mixture_width
        self.mixture_depth  = mixture_depth
        self.aug_severity   = aug_severity
        self.augmentations  = augmentations
        # fmt: on

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.

        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros([image.size[1], image.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)

# class AddNoise(nn.Module):
#     def __init__(self, s=0.005, p=0.9):
#         super().__init__()
#         assert isinstance(s, float) or (isinstance(p, float))  # 判断输入参数格式是否正确
#         self.s = 1 - s
#         self.p = p
#
#     def forward(self, img):
#         if random.uniform(0, 1) < self.p:
#             b, c, h, w = img.shape
#             signal_pct = self.s
#             noise_pct = (1 - self.s)
#
#             mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
#             mask = np.repeat(mask, c, axis=0)
#
#             img[:, mask == 1] = 255  # 白噪声
#             img[:, mask == 2] = 0  # 黑噪声
#             return img
#
#         else:
#             return img
