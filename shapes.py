import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture


def curve_to_efsd(z, harmonic_count):
    def setup():
        dx = np.diff(z[:, 0])
        dy = np.diff(z[:, 1])
        dt = np.sqrt(dx**2 + dy**2)
        t = np.concatenate(([0], np.cumsum(dt)))
        dx_cs = np.cumsum(dx)
        dy_cs = np.cumsum(dy)
        I = np.arange(1, harmonic_count)
        i, u = np.meshgrid(I, 2 * np.pi * t / t[-1])

        return t[-1], t, np.concatenate(([0], dx_cs)), np.concatenate(([0], dy_cs)), dx/dt, dy/dt, I, i*u

    T, t, dx_sums, dy_sums, dx_over_dt, dy_over_dt, I, grid = setup()

    # Calculate the series expansion.
    cos_term = np.cos(grid)
    sin_term = np.sin(grid)
    efsd = np.zeros((4, harmonic_count))
    efsd[0, 1:] = np.sum(dx_over_dt.reshape(-1, 1) * (cos_term[1:]-cos_term[:-1]), axis=0)
    efsd[1, 1:] = np.sum(dx_over_dt.reshape(-1, 1) * (sin_term[1:]-sin_term[:-1]), axis=0)
    efsd[2, 1:] = np.sum(dy_over_dt.reshape(-1, 1) * (cos_term[1:]-cos_term[:-1]), axis=0)
    efsd[3, 1:] = np.sum(dy_over_dt.reshape(-1, 1) * (sin_term[1:]-sin_term[:-1]), axis=0)
    efsd[:, 1:] *= T / (2 * np.pi**2 * I**2)

    # Calculate the DC components.
    tt1 = np.diff(t)
    tt2 = np.diff(t**2)
    a0 = 1 / T * np.sum(dx_over_dt/2*tt2 + (dx_sums[:-1]-dx_over_dt*t[:-1])*tt1)
    c0 = 1 / T * np.sum(dy_over_dt/2*tt2 + (dy_sums[:-1]-dy_over_dt*t[:-1])*tt1)
    efsd[0, 0] = a0 + z[0, 0]
    efsd[2, 0] = c0 + z[0, 1]
    return efsd


def efsd_to_curve(efsd, sample_count):
    n, t = np.meshgrid(np.arange(1, efsd.shape[-1]), np.linspace(0, 1, sample_count))

    grid = 2 * np.pi * n * t
    cos_term = np.cos(grid)
    sin_term = np.sin(grid)
    return np.vstack((
        efsd[0, 0] + np.sum(efsd[0, 1:]*cos_term + efsd[1, 1:]*sin_term, axis=-1),
        efsd[2, 0] + np.sum(efsd[2, 1:]*cos_term + efsd[3, 1:]*sin_term, axis=-1),
    )).T


class MaskGenerator:
    def __init__(self, contours, harmonic_count):
        descriptors = []
        for cc, nc in contours:
            descriptors.append(np.concatenate((
                curve_to_efsd(cc, harmonic_count).T.ravel(),
                curve_to_efsd(nc, harmonic_count).T.ravel(),
            )))
        mix = GaussianMixture(covariance_type='full', n_components=harmonic_count)
        self.pdf = mix.fit(np.array(descriptors))

    def __call__(self, resolution, padding=30):
        sample = self.pdf.sample()[0][0, :]
        half = len(sample) // 2
        cd = sample[:half].reshape(-1, 4).T
        nd = sample[half:].reshape(-1, 4).T
        z1 = efsd_to_curve(cd, resolution)
        z2 = efsd_to_curve(nd, resolution)

        lower = np.floor(np.min(z1, axis=0)).astype(int)
        upper = np.ceil(np.max(z1, axis=0)).astype(int)
        bounds = upper + lower + 2*padding

        im = np.zeros(bounds, dtype=np.uint8)
        cv.drawContours(im, [z1.astype(int)], -1, 128, cv.FILLED, offset=lower+padding)
        cv.drawContours(im, [z2.astype(int)], -1, 255, cv.FILLED, offset=lower+padding)
        return im
