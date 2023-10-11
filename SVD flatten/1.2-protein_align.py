from skimage import io
import numpy as np
import os
pwd = os.path.dirname(__file__)
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt

debug = 0
save = 0
folder = pwd
sample = 'example data'
img_stack = io.imread(os.path.join(folder, sample, '1-output_1.tif'))
mask_stack = io.imread(os.path.join(folder, sample, '0-mask.tif')).astype('bool')
outlier_stack = np.zeros_like(mask_stack)

diff = np.zeros(img_stack.shape[0]-1)
for i in np.arange(img_stack.shape[0]-1):
    img_1 = img_stack[i]
    mask_1 = mask_stack[i]
    img_2 = img_stack[i+1]
    mask_2 = mask_stack[i+1]

    diff[i] = np.median(img_2[mask_2]) - np.median(img_1[mask_1])
drift = np.cumsum(diff)

img_align = img_stack.copy()
for i in np.arange(img_stack.shape[0]-1):
    img_align[i+1] -= drift[i]

bgd = img_align[~mask_stack * ~outlier_stack].flatten()
bgd_n, bgd_bins = np.histogram(bgd, bins=256)
bgd_bins = .5*(bgd_bins[1:]+bgd_bins[:-1])

bgd_mod = GaussianModel()
bgd_parms = bgd_mod.guess(bgd_n, x=bgd_bins)
bgd_parms['center'].value = -2.5
bgd_out = bgd_mod.fit(bgd_n, bgd_parms, x=bgd_bins)
if debug:
    plt.figure()
    plt.bar(bgd_bins, bgd_n, width=np.mean(np.diff(bgd_bins)))
    plt.plot(bgd_bins, bgd_out.best_fit)
    plt.show()

bgd_x = bgd_out.params['center'].value

img_align -= bgd_x

plt.figure()
plt.hist(img_stack.flatten(), bins=256, label='raw', alpha=.5)
plt.hist(img_align.flatten(), bins=256, label='aligned', alpha=.5)
plt.legend()
plt.show()

if save:
    io.imsave(os.path.join(folder, sample, '1-output_2.tif'), img_align)
