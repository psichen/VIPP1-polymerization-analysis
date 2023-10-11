from skimage import io
import os
pwd = os.path.dirname(__file__)
import numpy as np

folder = pwd
sample = 'example data'
img_stack = io.imread(os.path.join(folder, sample, 'input.tif'))
mask_stack = io.imread(os.path.join(folder, sample, '0-mask.tif')).astype('bool')
outlier_stack = np.zeros_like(mask_stack)

def displace(img, mask, outlier):
    displace = np.zeros_like(img)
    for i in np.arange(img.shape[0]):
        outlier_line = outlier[i,:]
        mask_line = mask[i,:]

        if np.sum(mask_line) and np.sum(mask_line)!=mask_line.shape[0]:
            if np.sum(~mask_line * ~outlier_line):
                bgd_height = np.median(img[i, :][~mask_line * ~outlier_line])
                ptn_height = np.median(img[i, :][mask_line])
                displace[i, mask_line] = ptn_height - bgd_height

    displace_img = np.median(displace[displace!=0])
    displace[mask] = displace_img
    return displace

def line_median(img, img_displace):
    img_med = np.zeros_like(img)
    for i in np.arange(img.shape[0]):
        img_med[i,:] = img[i,:] - np.median(img_displace[i,:])

    return img_med

displace_stack = np.zeros_like(img_stack)
for i in np.arange(img_stack.shape[0]):
    displace_stack[i] = displace(img_stack[i], mask_stack[i],outlier_stack[i])

img_med_stack = np.zeros_like(img_stack)
for i in np.arange(img_stack.shape[0]):
    img_med_stack[i] = line_median(img_stack[i], img_stack[i] - displace_stack[i])

io.imsave(os.path.join(folder, sample, '1-output_1.tif'), img_med_stack)
