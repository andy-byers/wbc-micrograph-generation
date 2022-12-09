#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import namedtuple
import sys

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.utils import save_image


def process_image(im):
    # Convert between RGB and BGR.
    return torch.flip(im, [0])


def unprocess_image(im):
    im = im.cpu().detach()
    im = torch.clip(im, 0, 1)
    return process_image(im)


def filter_features(features, targets):
    return [f for f, target in zip(features, features._fields) if target in targets]


# Modified from https://gitlab.com/vitadx/articles/generic_isolated_cell_images_generator
def normalize_patches(patches):
    norm = torch.sqrt(torch.sum(torch.square(patches), dim=(1, 2, 3)))
    return torch.reshape(norm, (norm.shape[0], 1, 1))


# Modified from https://gitlab.com/vitadx/articles/generic_isolated_cell_images_generator
def extract_patches(im, patch_shape, stride=1, shuffled=False):
    patches = (im.unfold(1, patch_shape, stride)
                 .unfold(2, patch_shape, stride))
    patches = torch.permute(patches, (1, 2, 0, 3, 4))
    count = patches.shape[0] * patches.shape[1]
    patches = torch.reshape(patches, (count, -1, patch_shape, patch_shape))

    norm = normalize_patches(patches)
    norm = torch.reshape(norm, (count, 1, 1, 1))

    if shuffled:
        shuffle = torch.randperm(count)
        patches = patches[shuffle]
        norm = norm[shuffle]
    return patches, norm


# Modified from https://gitlab.com/vitadx/articles/generic_isolated_cell_images_generator
def find_neighbor_patches(p, q):
    p = torch.reshape(p, (p.shape[0], -1))
    q = torch.reshape(q, (q.shape[0], -1))
    return torch.argmax(torch.tensordot(p, q.T, dims=1), dim=-1).long()


def cat_feats(feats, masks):
    return [torch.cat((y, m), dim=0) for y, m in zip(feats, masks)]


def total_mrf_loss(alphas, t_patches, t_norms, s_patches, s_norms):
    scale = 1 / np.sum(alphas)
    loss = 0.0
    for alpha, t_patch, tn, s_patch, sn in zip(alphas, t_patches, t_norms, s_patches, s_norms):
        indices = find_neighbor_patches(t_patch / tn, s_patch / sn)
        loss += alpha * scale * F.mse_loss(t_patch, s_patch[indices])
    return loss


# Modified from https://github.com/jxgu1016/Total_Variation_Loss.pytorch/blob/master/TVLoss.py
def tv_loss(im):
    c, h, w = im.shape
    tv_y = torch.sum(torch.square(torch.diff(im, dim=-2)))
    tv_x = torch.sum(torch.square(torch.diff(im, dim=-1)))
    return (tv_x+tv_y) / (c*h*w)


def expand_mask(mask, labels=None):
    """Spread semantic segmentation mask labels over channels."""
    labels = np.unique(mask) if not labels else labels
    fixed = np.zeros((*mask.shape, len(labels)), dtype=float)
    for i, u in enumerate(labels):
        fixed[..., i] = mask == u
    return fixed


def load_image(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return np.array(im, dtype=float) / 255.0


def load_mask(path):
    with Image.open(path) as im:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        # Convert each RGB pixel value into a unique integer.
        ar = np.array(im, dtype=np.uint8)
        ar = ar[..., 0] + ar[..., 1]*256 + ar[..., 2]*256**2
        return expand_mask(ar)


LAYER_NAMES = (
    'conv_1_1', 'conv_1_2',
    'conv_2_1', 'conv_2_2',
    'conv_3_1', 'conv_3_2',
    'conv_4_1', 'conv_4_2',
    'conv_5_1', 'conv_5_2',
)
LAYERS = namedtuple('vgg_layers', LAYER_NAMES)
VGG_MEAN = torch.tensor(([[0.485]], [[0.456]], [[0.406]]))
VGG_STD = torch.tensor(([[0.229]], [[0.224]], [[0.225]]))


# Modified from the PyTorch neural style transfer example.
class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)

    def forward(self, img):
        return (img-self.mean) / self.std


class Vgg19(nn.Module):
    def __init__(self, normalizer):
        super(Vgg19, self).__init__()
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1_1 = nn.Sequential(
            normalizer,
            model.features[0:2],
        )
        self.conv1_2 = model.features[2:4]
        self.conv2_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            model.features[5:7],
        )
        self.conv2_2 = model.features[7:9]
        self.conv3_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            model.features[10:12],
        )
        self.conv3_2 = model.features[12:14]
        self.conv4_1 = nn.Sequential(
            model.features[12:18],
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            model.features[19:21],
        )
        self.conv4_2 = model.features[21:23]
        self.conv5_1 = nn.Sequential(
            model.features[23:27],
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            model.features[28:30],
        )
        self.conv5_2 = model.features[30:32]

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)
        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv4_1 = self.conv4_1(conv3_2)
        conv4_2 = self.conv4_2(conv4_1)
        conv5_1 = self.conv5_1(conv4_2)
        conv5_2 = self.conv5_2(conv5_1)

        return LAYERS(
            conv1_1, conv1_2,
            conv2_1, conv2_2,
            conv3_1, conv3_2,
            conv4_1, conv4_2,
            conv5_1, conv5_2,
        )


class Downsampler(nn.Module):
    def __init__(self):
        super(Downsampler, self).__init__()
        same_pool = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        down_pool = {'kernel_size': 2, 'stride': 2, 'padding': 0}

        self.pool1_1 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
        )
        self.pool1_2 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
        )
        self.pool2_1 = nn.Sequential(
            nn.AvgPool2d(**down_pool),
            nn.AvgPool2d(**same_pool),
        )
        self.pool2_2 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
        )
        self.pool3_1 = nn.Sequential(
            nn.AvgPool2d(**down_pool),
            nn.AvgPool2d(**same_pool),
        )
        self.pool3_2 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
        )
        self.pool4_1 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
            nn.AvgPool2d(**same_pool),
            nn.AvgPool2d(**same_pool),
            nn.AvgPool2d(**down_pool),
            nn.AvgPool2d(**same_pool),
        )
        self.pool4_2 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
        )
        self.pool5_1 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
            nn.AvgPool2d(**same_pool),
            nn.AvgPool2d(**down_pool),
            nn.AvgPool2d(**same_pool),
        )
        self.pool5_2 = nn.Sequential(
            nn.AvgPool2d(**same_pool),
        )

    def forward(self, x):
        # Note that most likely only the poolings with stride 2 are necessary. TODO
        pool1_1 = self.pool1_1(x)
        pool1_2 = self.pool1_2(pool1_1)
        pool2_1 = self.pool2_1(pool1_2)
        pool2_2 = self.pool2_2(pool2_1)
        pool3_1 = self.pool3_1(pool2_2)
        pool3_2 = self.pool3_2(pool3_1)
        pool4_1 = self.pool4_1(pool3_2)
        pool4_2 = self.pool4_2(pool4_1)
        pool5_1 = self.pool5_1(pool4_2)
        pool5_2 = self.pool5_2(pool5_1)

        return LAYERS(
            pool1_1, pool1_2,
            pool2_1, pool2_2,
            pool3_1, pool3_2,
            pool4_1, pool4_2,
            pool5_1, pool5_2,
        )


DEFAULT_LAYERS = (
    'conv_1_2',
    'conv_2_2',
    'conv_3_2',
    'conv_4_2',
    'conv_5_2',
)
DEFAULT_STYLE_FRACTIONS = (6, 4, 3, 2, 1)


class DoodleTransfer:
    def __init__(self, args):
        self.device = args.device
        self.style_layers = args.style_layers
        self.semantic_weight = args.semantic_weight
        self.vgg_weights = [w for w, _ in zip(args.vgg_weights, self.style_layers)]
        self.shuffle_patches = args.shuffle_patches
        self.args = args

        # Define the loss functions.
        self.style_loss = lambda *x: args.style_weight * total_mrf_loss(*x)
        self.tv_loss = lambda *x: args.tv_weight * tv_loss(*x)

        transform = transforms.Compose((
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(args.image_shape),
        ))
        normalizer = Normalizer(VGG_MEAN.to(self.device), VGG_STD.to(self.device))
        self.model = Vgg19(normalizer).to(self.device)
        self.model.requires_grad_(False)
        self.downsampler = Downsampler().to(self.device)
        self.downsampler.requires_grad_(False)

        style_image = process_image(transform(args.style_image)).to(self.device)
        style_features = filter_features(self.model(style_image), self.style_layers)
        style_mask = transform(args.style_mask).to(self.device)
        content_mask = transform(args.content_mask).to(self.device)

        # Downsample each semantic map to be size-compatible with the desired VGG19 activations.
        scaled_style_masks = filter_features(self.downsampler(style_mask * self.semantic_weight), self.style_layers)
        scaled_content_masks = filter_features(self.downsampler(content_mask * self.semantic_weight), self.style_layers)
        style_semantic = cat_feats(style_features, scaled_style_masks)
        self.scaled_content_masks = scaled_content_masks

        # Save the style patches and norms. They are the same each optimization round.
        self.style_patches, self.style_norms = self._extract_all_patches(style_semantic)
        self.target_image = process_image(transform(args.target_image))
        self.target_image = torch.autograd.Variable(self.target_image, requires_grad=True)
        self.optimizer = optim.LBFGS([self.target_image])
        self.style_weight = args.style_weight

        if args.use_tensorboard:
            self.writer = SummaryWriter()
            self.writer.add_image('style-image', style_image.squeeze(0))
            self.writer.add_image('style-mask', style_mask.squeeze(0))
            self.writer.add_image('content-mask', content_mask.squeeze(0))
            self.writer.add_image('target-image', self.target_image.squeeze(0))
            self.iteration = 0

    @staticmethod
    def _iter_batches(features, masks, batch_size):
        fc = features.shape[0]
        mc = masks.shape[0]

        if mc >= batch_size:
            raise ValueError(f'number of semantic channels {mc} should be less than the batch size {batch_size}')

        feature_batch_size = batch_size - mc
        for i in range(0, fc, feature_batch_size):
            count = min(fc - i, feature_batch_size)
            feats = features[i:i+count]
            yield torch.cat((feats, masks), dim=0)

    def _extract_all_patches(self, ys):
        patches, norms = [], []
        for y in ys:
            patch, norm = extract_patches(y, 3, shuffled=self.shuffle_patches)
            patches.append(patch)
            norms.append(norm)
        return patches, norms

    def _step(self):
        self.optimizer.zero_grad(set_to_none=True)
        tx = self.target_image.to(self.device)
        ys = self.model(tx)

        target_features = filter_features(ys, self.style_layers)
        target_semantic = cat_feats(target_features, self.scaled_content_masks)
        target_patches, target_norms = self._extract_all_patches(target_semantic)
        style_loss = self.style_loss(self.vgg_weights, target_patches, target_norms,
                                     self.style_patches, self.style_norms)
        loss = self.style_weight*style_loss + self.tv_loss(tx)
        loss.backward()

        if self.args.use_tensorboard:
            self._write_report(style_loss, tv_loss)
        return loss

    def _write_report(self, style_loss, tv_loss):
        self.writer.add_scalar('Loss/style', style_loss, self.iteration)
        self.writer.add_scalar('Loss/tv', tv_loss, self.iteration)

        if self.iteration % self.args.detail_interval == 0:
            example = self.target
            self.writer.add_image('target-image',  example, self.iteration)
            self.writer.add_scalar('Stats/target-min', torch.min(example), self.iteration)
            self.writer.add_scalar('Stats/target-max', torch.max(example), self.iteration)
            self.writer.add_scalar('Stats/target-mean', torch.mean(example), self.iteration)
            self.writer.add_scalar('Stats/target-std', torch.std(example), self.iteration)
        self.iteration += 1

    @property
    def target(self):
        with torch.no_grad():
            return unprocess_image(self.target_image)

    def __call__(self, iterations=10):
        for i in range(iterations):
            self.optimizer.step(self._step)


def main():
    parser = ArgumentParser(description='Optimization-based neural style transfer')
    parser.add_argument('--output-image', type=str, help='Location to write the output image')
    parser.add_argument('--image-shape', type=int, nargs='+', help='Rescale images to this shape before optimization')
    parser.add_argument('--target-image', type=str, nargs='+', help='An image to optimize')
    parser.add_argument('--style-image', type=str, nargs='+', required=True,
                        help='An image from which to transfer style')
    parser.add_argument('--style-mask', type=str, nargs='+', required=True,
                        help='A semantic segmentation of the style image')
    parser.add_argument('--content-mask', type=str, nargs='+', required=True,
                        help='A semantic segmentation of the content image')
    parser.add_argument('--iterations', type=int, default=100, help='Number of optimization passes to run')
    parser.add_argument('--vgg-weights', type=float, nargs='+', default=DEFAULT_STYLE_FRACTIONS,
                        help='Importance of each style activation layer to the MRF loss computation')
    parser.add_argument('--keep-patch-order', action='store_true',
                        help='Do not shuffle neural patches before finding nearest neighbors')
    parser.add_argument('--semantic-weight', type=float, default=5_000, help='Importance of the semantic layers')
    parser.add_argument('--style-weight', type=float, default=150, help='Importance of the style loss')
    parser.add_argument('--tv-weight', type=float, default=1_000, help='Importance of the TV regularization')
    parser.add_argument('--device', default='cuda', help='Device on which to run the optimization')
    parser.add_argument('--use-tensorboard', action='store_true', help='Log statistics to tensorboard')
    parser.add_argument('--style-layers', nargs='+', default=DEFAULT_LAYERS,
                        help='Names of VGG19 feature maps to use for the style loss')
    args = parser.parse_args()

    try:
        for name in args.style_layers:
            if name not in LAYER_NAMES:
                raise ValueError(f'activation layer "{name}" does not exist in VGG19')

        args.style_image = load_image(' '.join(args.style_image))
        args.style_mask = load_mask(' '.join(args.style_mask))
        args.content_mask = load_mask(' '.join(args.content_mask))
        args.shuffle_patches = not args.keep_patch_order

        if not args.image_shape:
            args.image_shape = args.content_mask.shape[:-1]
        if len(args.image_shape) == 1:
            args.image_shape *= 2
        if args.target_image:
            args.target_image = load_image(' '.join(args.target_image))
        else:
            args.target_image = np.random.random(args.image_shape + [3])

        if not len(args.style_layers):
            raise ValueError('at least 1 style activation must be used')

        stylizer = DoodleTransfer(args)
        stylizer(args.iterations)
        target = stylizer.target

        if args.output_image:
            save_image(target, args.output_image)
        else:
            target = torch.permute(target, (1, 2, 0))
            plt.imshow(target)
            plt.show()

    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
