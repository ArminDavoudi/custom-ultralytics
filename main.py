import re
import os
import cv2
import math
import mlflow
import platform
import numpy as np
import seaborn as sns
from tqdm import tqdm
import logging as LOGGER
from importlib import metadata
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("custom-yolo-classification")

# Defaults, based on yolo default.yaml file
IMGSZ = 640
DEFAULT_MEAN = (0, 0, 0)
DEFAULT_STD = (1, 1, 1)
DEFAULT_CROP_FRACTION = 1.0
SCALE = (0.5, 1.0)
RATIO = None
AUTO_AUGMENT = "randaugment"
TORCHVISION_VERSION = metadata.version("torchvision")
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
TRAIN_PATH = 'datasets/apples/train'
VAL_PATH = 'datasets/apples/val'
TEST_PATH = 'datasets/apples/test'
NUM_WORKERS = 4
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-3
LR_FACTOR = 0.02
MODEL_SAVE_PATH = '.'


class ClassifyLetterBox:
    """
    A class for resizing and padding images for classification tasks.

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).
    It resizes and pads images to a specified size while maintaining the original aspect ratio.

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically calculates the short side using stride.
        stride (int): The stride value, used when 'auto' is True.

    Methods:
        __call__: Applies the letterbox transformation to an input image.

    Examples:
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size=(IMGSZ, IMGSZ), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox object for image preprocessing.

        This class is designed to be part of a transformation pipeline for image classification tasks. It resizes and
        pads images to a specified size while maintaining the original aspect ratio.

        Args:
            size (int | Tuple[int, int]): Target size for the letterboxed image. If an int, a square image of
                (size, size) is created. If a tuple, it should be (height, width).
            auto (bool): If True, automatically calculates the short side based on stride. Default is False.
            stride (int): The stride value, used when 'auto' is True. Default is 32.

        Attributes:
            h (int): Target height of the letterboxed image.
            w (int): Target width of the letterboxed image.
            auto (bool): Flag indicating whether to automatically calculate short side.
            stride (int): Stride value for automatic short side calculation.

        Examples:
            >>> transform = ClassifyLetterBox(size=224)
            >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> result = transform(img)
            >>> print(result.shape)
            (224, 224, 3)
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads an image using the letterbox method.

        This method resizes the input image to fit within the specified dimensions while maintaining its aspect ratio,
        then pads the resized image to match the target size.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C).

        Returns:
            (numpy.ndarray): Resized and padded image as a numpy array with shape (hs, ws, 3), where hs and ws are
                the target height and width respectively.

        Examples:
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # Calculate padding dimensions
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # Create padded image
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top: top + h, left: left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x


class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network."""

    def __init__(self, c4_channels, c6_channels, c8_channels, out_channels):
        super().__init__()

        # Lateral connections (1x1 convs to match channels)
        self.lat_c4 = Conv(c4_channels, out_channels, k=1)
        self.lat_c6 = Conv(c6_channels, out_channels, k=1)
        self.lat_c8 = Conv(c8_channels, out_channels, k=1)

        # Top-down pathway
        self.td_c6 = Conv(out_channels, out_channels, k=3, p=1)
        self.td_c4 = Conv(out_channels, out_channels, k=3, p=1)

        # Bottom-up pathway
        self.bu_c6 = Conv(out_channels, out_channels, k=3, p=1)
        self.bu_c8 = Conv(out_channels, out_channels, k=3, p=1)

        # Upsample and Downsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Weighted features fusion
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        # Final convolution to reduce channels
        self.final_conv = Conv(out_channels * 3, out_channels, k=1)

    def _weighted_fusion(self, x, weights):
        """Applies weighted fusion to the input features."""
        w = F.relu(weights)
        w_sum = w.sum() + self.epsilon
        w_normalized = w / w_sum

        out = 0
        for i, feat in enumerate(x):
            out += feat * w_normalized[i]
        return out

    def forward(self, c4, c6, c8):
        # Lateral connections
        p4 = self.lat_c4(c4)
        p6 = self.lat_c6(c6)
        p8 = self.lat_c8(c8)

        # Top-down pathway
        p8_td = p8
        p6_td = self._weighted_fusion(
            [p6, self.upsample(p8_td)],
            self.w1
        )
        p6_td = self.td_c6(p6_td)

        p4_td = self._weighted_fusion(
            [p4, self.upsample(p6_td)],
            self.w1
        )
        p4_td = self.td_c4(p4_td)

        # Bottom-up pathway
        p4_out = p4_td
        p6_out = self._weighted_fusion(
            [p6_td, self.downsample(p4_out)],
            self.w1
        )
        p6_out = self.bu_c6(p6_out)

        p8_out = self._weighted_fusion(
            [p8_td, self.downsample(p6_out)],
            self.w1
        )
        p8_out = self.bu_c8(p8_out)

        # Final fusion
        # Downsample p4_out and p6_out to match p8_out's spatial dimensions
        p4_out_downsampled = F.max_pool2d(p4_out, kernel_size=4, stride=4)  # Reduce (512, 80, 80) -> (512, 20, 20)
        p6_out_downsampled = F.max_pool2d(p6_out, kernel_size=2, stride=2)  # Reduce (512, 40, 40) -> (512, 20, 20)

        # Concatenate all three feature maps along the channel dimension
        fused_features = torch.cat([p4_out_downsampled, p6_out_downsampled, p8_out], dim=1)  # Shape: (3*512, 20, 20)

        # Apply final convolution to reduce channels
        final_features = self.final_conv(fused_features)

        return final_features


class CustomYOLO(nn.Module):
    def __init__(self, scales=[0.33, 0.50, 2, 1024], nc=2):
        super().__init__()
        d, w, r, max_ch = scales  # Change scales based on yolo-n-s-m-l values

        # ========== YOLO model blocks ========== #
        # Backbone
        self.c0 = Conv(c1=3, c2=round(64 * w), k=3, s=2, p=1)
        self.c1 = Conv(c1=round(64 * w), c2=round(128 * w), k=3, s=2, p=1)
        self.cf2 = C2f(c1=round(128 * w), c2=round(128 * w), n=round(3 * d), shortcut=True)
        self.c3 = Conv(c1=round(128 * w), c2=round(256 * w), k=3, s=2, p=1)
        self.cf4 = C2f(c1=round(256 * w), c2=round(256 * w), n=round(6 * d), shortcut=True)
        self.c5 = Conv(c1=round(256 * w), c2=round(512 * w), k=3, s=2, p=1)
        self.cf6 = C2f(c1=round(512 * w), c2=round(512 * w), n=round(6 * d), shortcut=True)
        self.c7 = Conv(c1=round(512 * w), c2=round(512 * w * r), k=3, s=2, p=1)
        self.cf8 = C2f(c1=round(512 * w * r), c2=round(512 * w * r), n=round(3 * d), shortcut=True)

        # BiFPN head
        self.bifpn = BiFPN(
            c4_channels=round(256 * w),         # cf4 channels
            c6_channels=round(512 * w),         # cf6 channels
            c8_channels=round(512 * w * r),     # cf8 channels
            out_channels=round(512 * w)         # output channels
        )

        # Classifier
        self.cls = Classify(c1=round(512 * w), c2=nc)

    def forward(self, x):
        # Initial convolutions
        x = self.c0(x)
        x = self.c1(x)
        x = self.cf2(x)

        # First feature map (cf4)
        x = self.c3(x)
        cf4_out = self.cf4(x)

        # Second feature map (cf6)
        x = self.c5(cf4_out)
        cf6_out = self.cf6(x)

        # Third feature map (cf8)
        x = self.c7(cf6_out)
        cf8_out = self.cf8(x)

        # BiFPN
        fpn_out = self.bifpn(cf4_out, cf6_out, cf8_out)

        # Classification
        return self.cls(fpn_out)


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.
        msg (str): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Examples:
        Check if current version is exactly 22.04
        >>> check_version(current="22.04", required="==22.04")

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04
        >>> check_version(current="22.04", required="<=22.04")

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"WARNING ⚠️ {current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            LOGGER.warning(warning)
    return result


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def classify_transforms(
    size=IMGSZ,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation="BILINEAR",
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Creates a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Fraction of the image to be cropped.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl.extend(
        [
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return T.Compose(tfl)


def classify_augmentations(
    size=IMGSZ,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=SCALE,
    ratio=RATIO,
    hflip=0.5,
    vflip=0.0,
    auto_augment=AUTO_AUGMENT,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation="BILINEAR",
):
    """
    Creates a composition of image augmentation transforms for classification tasks.

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.

    Args:
        size (int): Target size for the image after transformations.
        mean (tuple): Mean values for normalization, one per channel.
        std (tuple): Standard deviation values for normalization, one per channel.
        scale (tuple | None): Range of size of the origin size cropped.
        ratio (tuple | None): Range of aspect ratio of the origin aspect ratio cropped.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_v (float): Image HSV-Value augmentation factor.
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        erasing (float): Probability of random erasing.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    Returns:
        (torchvision.transforms.Compose): A composition of image augmentation transforms.

    Examples:
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if check_version(TORCHVISION_VERSION, "0.11.0"):
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            if check_version(TORCHVISION_VERSION, "0.13.0"):
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            if check_version(TORCHVISION_VERSION, "0.10.0"):
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_logits = []

    with tqdm(dataloader, desc="Training", unit="batch") as train_bar:
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(outputs.cpu().detach().numpy())

            # Update progress bar
            train_bar.set_postfix(loss=total_loss / total, accuracy=correct / total)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, all_labels, all_preds, all_logits


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_logits = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation", unit="batch") as val_bar:
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Metrics
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_logits.extend(outputs.cpu().detach().numpy())

                # Update progress bar
                val_bar.set_postfix(loss=total_loss / total, accuracy=correct / total)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, all_labels, all_preds, all_logits


def log_metrics(loss, accuracy, labels, preds, logits, epoch=1, phase="train", plots=False):
    # Calculate f1, precision, and recall
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted", zero_division=1)
    recall = recall_score(labels, preds, average="weighted", zero_division=1)

    print(f"{phase} loss: {loss:.4f} | {phase} accuracy: {accuracy:.4f} | {phase} f1 score: {f1:.4f}")

    mlflow.log_metrics({
        f"{phase}_loss": loss,
        f"{phase}_accuracy": accuracy,
        f"{phase}_f1": f1,
        f"{phase}_precision": precision,
        f"{phase}_recall": recall
    }, step=epoch)

    if plots:
        # Compute Confusion Matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"{phase.capitalize()} Confusion Matrix")
        mlflow.log_figure(plt.gcf(), f"{phase}/confusion_matrix.png")
        plt.close()

        # Compute predicted probabilities and ROC Curve
        probs = torch.nn.functional.softmax(torch.tensor(np.array(logits)), dim=1).numpy()
        y_true = np.array(labels)
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])  # Take probabilities for class 1
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{phase.capitalize()} Receiver Operating Characteristic Curve")
        plt.legend()
        mlflow.log_figure(plt.gcf(), f"{phase}/roc_curve.png")
        plt.close()

        # Compute Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, probs[:, 1])
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="green", lw=2, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{phase.capitalize()} Precision-Recall Curve")
        plt.legend()
        mlflow.log_figure(plt.gcf(), f"{phase}/pr_curve.png")
        plt.close()

        # Plot probability distribution
        plt.figure(figsize=(6, 5))
        plt.hist(probs[:, 1][y_true == 0], bins=20, alpha=0.5, label="Class 0", color="blue")
        plt.hist(probs[:, 1][y_true == 1], bins=20, alpha=0.5, label="Class 1", color="red")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"{phase.capitalize()} Predicted Probabilities Distribution")
        plt.legend()
        mlflow.log_figure(plt.gcf(), f"{phase}/probability_distribution.png")
        plt.close()


def get_transform(phase: str = 'train'):
    if phase == 'train':
        return classify_augmentations()
    return classify_transforms()


def get_model_info(model, imgsz):
    """
    Returns the model, input image size, number of parameters, and GFLOPs.

    Args:
        model (torch.nn.Module): The model to analyze.
        imgsz (int or tuple): The input image size (e.g., 640 or (3, 640, 640)).

    Returns:
        dict: A dictionary containing the model, input image size, number of parameters, and GFLOPs.
    """
    if isinstance(imgsz, int):
        imgsz = (3, imgsz, imgsz)  # Assume 3 channels (RGB) if imgsz is an integer

    # Calculate model summary
    model_stats = summary(model, input_size=(1, *imgsz), verbose=0)

    # Extract number of parameters and GFLOPs
    num_params = model_stats.total_params
    gflops = model_stats.total_mult_adds / 1e9  # Convert FLOPs to GFLOPs

    print(f"Model info: Trainable parameters: {num_params}, {gflops:.2f} GFLOPs \n")


# TODO Contrastive learning (dataloaders, loss)
# ==================================================================================================================== #
train_set = ImageFolder(TRAIN_PATH, get_transform('train'))
val_set = ImageFolder(VAL_PATH, get_transform('validation'))
test_set = ImageFolder(TEST_PATH, get_transform('validation'))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomYOLO(nc=2).to(device)
get_model_info(model, imgsz=IMGSZ)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.937, weight_decay=5e-4)
scheduler = LambdaLR(optimizer, lr_lambda=lambda x: max(1 - x / NUM_EPOCHS, 0) * (1.0 - LR_FACTOR) + LR_FACTOR)
best_val_f1 = 0.0

# Training loop
with mlflow.start_run():
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        train_loss, train_accuracy, train_labels, train_preds, train_logits = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_accuracy, val_labels, val_preds, val_logits = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Metrics and MLFLOW logs
        log_metrics(train_loss, train_accuracy, train_labels, train_preds, train_logits, epoch, phase='train')
        log_metrics(val_loss, val_accuracy, val_labels, val_preds, val_logits, epoch, phase='validation')

        # Save the best model if F1 score improves
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best.pth"))
            print(f"New best model saved with F1 score: {best_val_f1:.4f}")

    # Final validation on test dataset
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "best.pth")))
    test_loss, test_accuracy, test_labels, test_preds, test_logits = validate(model, test_loader, criterion, device)
    log_metrics(test_loss, test_accuracy, test_labels, test_preds, test_logits, phase='test', plots=True)
