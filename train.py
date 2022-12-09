from argparse import ArgumentParser
import pickle
import sys

from dataset import SegmentationWBC
from shapes import MaskGenerator


def main():
    parser = ArgumentParser(description='Train a WBC segmentation mask generation model')
    parser.add_argument('--output', required=True, type=str, help='Location to write the trained model')
    parser.add_argument('--types', nargs='+', default=['monocyte'], help='WBC types to learn morphologies from')
    parser.add_argument('--harmonics', type=int, default=64, help='Number of harmonics')
    parser.add_argument('--samples', type=int, default=64, help='Number of samples to take along each source contour')

    try:
        args = parser.parse_args()
        wbc = SegmentationWBC('.', args.samples)
        contours = []

        wbc_type_names = ('neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil')
        for t in args.types:
            if t not in wbc_type_names:
                raise ValueError(f'"{t}" is not a valid WBC type')
            contours += wbc.fetch_contours(wbc_type_names.index(t) + 1)

        if len(contours) <= args.harmonics:
            raise ValueError(f'number of examples {len(contours)} must be exceed the number of harmonics {args.harmonics}')
        generator = MaskGenerator(contours, args.harmonics)

        with open(args.output, 'wb') as f:
            pickle.dump(generator, f)
        return 0

    except Exception as e:
        print(e, file=sys.stderr)
        raise
        return 1


if __name__ == '__main__':
    sys.exit(main())
