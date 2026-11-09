"""Repair the one-pixel coloured rim LibRaw sometimes leaves on the frame edge.

Some Bayer layouts have no sample for one of the colour planes in the outermost
column, and the demosaic (DHT/AHD alike) cannot interpolate it: that column comes
back with the missing channel near zero and the other two clipped near white.
After the film curve it reads as a 1px cyan/blue line down the right edge of the
photo, which only becomes obvious once the canvas is zoomed in far enough for a
single source pixel to cover several screen pixels.

It is not a lens or colour-management problem -- it is in the decoded pixels, so
it has to be fixed before the preview downscale, the LUT and the export all see
it.  We only touch an edge that is actually broken; a healthy frame is returned
byte-for-byte unchanged, so the calibrated colour references are unaffected.
"""

import numpy as np


def repair_demosaic_border(rgb):
    """Replace an outermost row/column with its neighbour when it is corrupted."""
    if (rgb is None or not isinstance(rgb, np.ndarray) or rgb.ndim != 3
            or rgb.shape[0] < 4 or rgb.shape[1] < 4):
        return rgb

    def broken(edge, ref):
        e = np.asarray(edge, dtype=np.float32)
        r = np.asarray(ref, dtype=np.float32)
        # A real frame edge can differ from its neighbour, but a whole column
        # diverging by ~30% of the local level on average is a demosaic failure,
        # not scene content (the failing column measured 3x its neighbour).
        return float(np.abs(e - r).mean()) > 0.30 * max(float(r.mean()), 1.0)

    if broken(rgb[:, -1, :], rgb[:, -2, :]):
        rgb[:, -1, :] = rgb[:, -2, :]
    if broken(rgb[:, 0, :], rgb[:, 1, :]):
        rgb[:, 0, :] = rgb[:, 1, :]
    if broken(rgb[-1, :, :], rgb[-2, :, :]):
        rgb[-1, :, :] = rgb[-2, :, :]
    if broken(rgb[0, :, :], rgb[1, :, :]):
        rgb[0, :, :] = rgb[1, :, :]
    return rgb
