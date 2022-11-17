"""
Major functions

Copyright (c) 2020 University of Illinois at Urbana-Champaign
(see LICENSE for details)
Written by Jindou Shi
"""

import numpy as np
import matplotlib as mpl
import matplotlib.colors
from skimage import exposure

from . import colors


def stain_SLAM(im_2, im_3, im_s, im_t, brightness_adjust=[2, 4, 0.5, 0.8]):
    # input: 4 2d array  (0,1)
    # 1. adjust brightness
    # 2.
    # output: RGB

    yellow = colors.select_cmap('yhot')
    cyan = colors.select_cmap('chot')
    green = colors.select_cmap('ghot')
    magn = colors.select_cmap('mhot')

    ym = mpl.colors.ListedColormap(yellow / 255.)
    cm = mpl.colors.ListedColormap(cyan / 255.)
    gm = mpl.colors.ListedColormap(green / 255.)
    mm = mpl.colors.ListedColormap(magn / 255.)

    imc_2 = ym(im_2 * brightness_adjust[0])[:, :, :3]
    imc_3 = cm(im_3 * brightness_adjust[1])[:, :, :3]
    imc_s = gm(im_s * brightness_adjust[2])[:, :, :3]
    imc_t = mm(im_t * brightness_adjust[3])[:, :, :3]

    imc = imc_2 + imc_3 + imc_s + imc_t
    imc[imc > 1] = 1.

    return imc


def colorize_image2D(c_name, img):
    """"""
    cmap_dict = {'2pf': 'yellowhot', '3pf': 'cyanhot', 'shg': 'greenhot', 'thg': 'magnettahot'}
    assert c_name.lower() in cmap_dict.keys(), '[Error] unknown channel name'
    cmap_name = cmap_dict[c_name.lower()]
    cmap_array = colors.select_cmap(cmap_name)


def norm_img2D(img, norm_mode, thres=None):
    """
    Normalize 2d images
    Reference https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html

    :param img: (2d array) input image
    :param norm_mode:   0: '0-1'
                        1: 'Adaptive equalization'
                        2: 'Contrast stretching'
                        3: 'Histogram equalization'
                        4: 'fix_threshold'
    :param thres: ()
    :return img_new: (2d array) normalized image
    """
    if img.max() != 1. or img.min() != 0.:
        vmax = np.max(img)
        vmin = np.min(img)
        img = (img - vmin) / (vmax - vmin)

    if norm_mode == 1:
        img_new = exposure.equalize_adapthist(img, clip_limit=0.03)
    elif norm_mode == 2:
        p2, p98 = np.percentile(img, (2, 98))
        img_new = exposure.rescale_intensity(img, in_range=(p2, p98))
    elif norm_mode == 3:
        img_new = exposure.equalize_hist(img)
    elif norm_mode == 4:
        img_new = img
    else:
        img_new = img
    return img_new














