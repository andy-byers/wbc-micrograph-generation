import os

import cv2 as cv
import numpy as np
from PIL import Image


class SegmentationWBC:
    def __init__(self, root, contour_length=64):
        root_fmt = os.path.join(root, 'segmentation_WBC')
        data_fmt = os.path.join(root_fmt,  'Dataset {}')
        type_fmt = os.path.join(root_fmt, 'Class Labels of Dataset {}.csv')
        cells, masks, types = [], [], []

        # Extract and combine entries from each of the two datasets.
        for i in range(2):
            data_dir = data_fmt.format(i + 1)
            type_file = type_fmt.format(i + 1)
            for path in sorted(os.listdir(data_dir)):
                with Image.open(os.path.join(data_dir, path)) as im:
                    # The cell images are stored as BMPs and the segmentation masks as PNGs.
                    (cells if path.endswith('.bmp') else masks).append(np.array(im))
            # Parse the CSV containing the WBC types.
            types.append(np.genfromtxt(type_file, skip_header=True, delimiter=',', dtype=int)[:, 1])

        for index, (mask, cell) in enumerate(zip(masks, cells)):
            nonzero = np.vstack(np.where(mask != 0)).T
            bounds0 = np.min(nonzero, axis=0)
            bounds1 = np.max(nonzero, axis=0)
            masks[index] = mask[bounds0[0]:bounds1[0]+1, bounds0[1]:bounds1[1]+1]
            cells[index] = cell[bounds0[0]:bounds1[0]+1, bounds0[1]:bounds1[1]+1]

        self.types = []
        self.masks = []
        self.cells = []
        self.contours = []

        for i, wbc_type in enumerate(np.concatenate((types[0], types[1]))):
            cc, nc = self.process_mask(masks[i])
            # For simplicity, we're only considering WBCs with one distinct
            # nuclear body.
            if len(nc) == 1:
                self.contours.append((
                    self.downsample_contour(np.squeeze(cc[0]), contour_length),
                    self.downsample_contour(np.squeeze(nc[0]), contour_length),
                ))
                self.types.append(wbc_type)
                self.masks.append(masks[i])
                self.cells.append(cells[i])

    def fetch_contours(self, wbc_type):
        return self._fetch(self.contours, wbc_type)

    def fetch_cells(self, wbc_type):
        return self._fetch(self.cells, wbc_type)

    def fetch_masks(self, wbc_type):
        return self._fetch(self.masks, wbc_type)

    def _fetch(self, target, wbc_type):
        return [target[i] for i, t in enumerate(self.types) if t == wbc_type]

    @staticmethod
    def downsample_contour(pt, length):
        assert length < pt.shape[0]
        # Make sure the curve is closed.
        if not np.allclose(pt[0], pt[-1]):
            pt = np.append(pt, [pt[0, :]], axis=0)
        target = np.linspace(0, 1, length)
        source = np.linspace(0, 1, pt.shape[0])
        return np.vstack((
            np.interp(target, source, pt[:, 0]),
            np.interp(target, source, pt[:, 1]),
        )).T

    @staticmethod
    def process_mask(mask):
        m1 = (mask == 128).astype(np.uint8) * 255
        m2 = (mask == 255).astype(np.uint8) * 255

        # Fill in where m2 was in m1.
        m1[m2 == 255] = 255

        # Get the exact boundary pixels from each mask. findContours() doesn't
        # produce points that are equally-spaced along the arc length, so we'll
        # just resample later.
        c1, *rest = cv.findContours(m1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        c2, *rest = cv.findContours(m2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        return c1, c2
