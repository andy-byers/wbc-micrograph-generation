from argparse import ArgumentParser
import pickle
import random
import sys

from torchvision import transforms

wbc_type_names = ('neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil')


def choose_style(wbc, style_type):
    style_images = wbc.fetch_cells(wbc_type_names.index(style_type))
    style_masks = wbc.fetch_masks(wbc_type_names.index(style_type))

    assert len(style_images) == len(style_masks)
    index = random.randint(0, len(style_images))
    return style_images[index], style_masks[index]


def main():
    parser = ArgumentParser(description='Generate an artificial WBC semantic segmentation map')
    parser.add_argument('--output-prefix', help='Path prefix to use when writing the generated masks to disk')
    parser.add_argument('--count', type=int, default=1, help='Number of masks to generate')
    parser.add_argument('--model', help='Location of a pickeled mask generation model')
    parser.add_argument('--resolution', type=int, default=64, help='Smoothness of generated contours')

    try:
        args = parser.parse_args()

        transform = transforms.ToPILImage()

        with open(args.model, 'rb') as f:
            generator = pickle.load(f)

        for i in range(args.count):
            mask = transform(generator(args.resolution))
            mask.save(f'{args.output_prefix}{str(i)}.png')

        return 0

    except Exception as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
