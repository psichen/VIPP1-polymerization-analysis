from skimage import io
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
import numpy as np
import os
pwd = os.path.dirname(__file__)
from lmfit.models import GaussianModel

save = 0
folder = pwd
sample = 'example data'
if 'cal_list' in locals():
    del cal_list
if 'aligned_list' in locals():
    del aligned_list

img_list = io.imread(os.path.join(folder, sample, 'SI_mov_flatten_2.tif'))

def align_frames(img_list):
    thres = threshold_otsu(img_list)
    mask_list = img_list > thres

    # align each frame to be constant protein height
    for img, mask in zip(img_list, mask_list):

        ptn = img.flatten()[mask.flatten()] # protein layer
        bgd = img.flatten()[~mask.flatten()] # background layer

        ptn_n, ptn_bins = np.histogram(ptn, bins=256, range=(img_list.min(), img_list.max()))
        ptn_bins = .5*(ptn_bins[1:]+ptn_bins[:-1])

        bgd_n, bgd_bins = np.histogram(bgd, bins=256, range=(img_list.min(), img_list.max()))
        bgd_bins = .5*(bgd_bins[1:]+bgd_bins[:-1])

        ptn_mod = GaussianModel()
        ptn_parms = ptn_mod.guess(ptn_n, x=ptn_bins)
        ptn_out = ptn_mod.fit(ptn_n, ptn_parms, x=ptn_bins)
        ptn_x = ptn_out.params['center'].value

        bgd_mod = GaussianModel()
        bgd_parms = bgd_mod.guess(bgd_n, x=bgd_bins)
        bgd_out = bgd_mod.fit(bgd_n, bgd_parms, x=bgd_bins)
        bgd_x = bgd_out.params['center'].value

        aligned = img - ptn_x
        if 'aligned_list' in locals():
            aligned_list = np.concatenate((aligned_list, [aligned]), axis=0)
        else:
            aligned_list = [aligned]

    if not isinstance(aligned_list, np.ndarray):
        aligned_list = np.array(aligned_list)

    bgd = aligned_list.flatten()[~mask_list.flatten()]
    bgd_n, bgd_bins = np.histogram(bgd, bins=256, range=(img_list.min(), img_list.max()))
    bgd_bins = .5*(bgd_bins[1:]+bgd_bins[:-1])
    bgd_mod = GaussianModel()
    bgd_parms = bgd_mod.guess(bgd_n, x=bgd_bins)
    bgd_out = bgd_mod.fit(bgd_n, bgd_parms, x=bgd_bins)
    bgd_x = bgd_out.params['center'].value
    aligned_list -= bgd_x

    return aligned_list

aligned_list = align_frames(img_list)

# displace between protein and background
thres = threshold_otsu(aligned_list)
mask_list = aligned_list > thres

ptn = aligned_list.flatten()[mask_list.flatten()] # protein layer
bgd = aligned_list.flatten()[~mask_list.flatten()] # background layer

ptn_n, ptn_bins = np.histogram(ptn, bins=256, range=(aligned_list.min(), aligned_list.max()))
ptn_bins = .5*(ptn_bins[1:]+ptn_bins[:-1])

bgd_n, bgd_bins = np.histogram(bgd, bins=256, range=(aligned_list.min(), aligned_list.max()))
bgd_bins = .5*(bgd_bins[1:]+bgd_bins[:-1])

ptn_mod = GaussianModel()
ptn_parms = ptn_mod.guess(ptn_n, x=ptn_bins)
ptn_out = ptn_mod.fit(ptn_n, ptn_parms, x=ptn_bins)
ptn_x = ptn_out.params['center'].value

bgd_mod = GaussianModel()
bgd_parms = bgd_mod.guess(bgd_n, x=bgd_bins)
bgd_out = bgd_mod.fit(bgd_n, bgd_parms, x=bgd_bins)
bgd_x = bgd_out.params['center'].value

displace = ptn_x - bgd_x

cal_list = aligned_list.flatten().copy()
cal_list[mask_list.flatten()] -= displace
cal_list = np.reshape(cal_list, (aligned_list.shape[0],aligned_list.shape[1]*aligned_list.shape[2]))

u, s, vh = np.linalg.svd(cal_list, full_matrices=False)
if u.shape[0] == u.shape[1]:
    smat = np.zeros_like(u)
elif vh.shape[0] == vh.shape[1]:
    smat = np.zeros_like(vh)
smat[0,0] = s[0]
bg_list = u @ smat @ vh
bg_list = np.reshape(bg_list, (aligned_list.shape[0], aligned_list.shape[1], aligned_list.shape[2]))

for aligned, bg in zip(aligned_list, bg_list):
    flatten = aligned - gaussian(bg, sigma=5)
    if 'flatten_list' in locals():
        flatten_list = np.concatenate((flatten_list, [flatten]), axis=0)
    else:
        flatten_list = [flatten]

# need 2 times of alignment to put background to zero
if not isinstance(flatten_list, np.ndarray):
    flatten_list = np.array(flatten_list)

flatten_list = align_frames(flatten_list)
flatten_list = align_frames(flatten_list)

if save:
    io.imsave(os.path.join(folder, sample, '2-output.tif'), flatten_list)
    print('saved!')
