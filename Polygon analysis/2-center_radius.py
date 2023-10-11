import os
pwd = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(os.path.dirname(pwd), 'sp_toolbox'))
import numpy as np
from skimage import io
from skimage import img_as_ubyte
from skimage.filters import apply_hysteresis_threshold
from skimage.segmentation import flood_fill
import napari

folder = pwd
i = 0 # index of data replicates
save = 0

th_h_ratio = .8 # ratio of maximum
th_l_ratio = 1.1 # ratio of median

def img_slice(path):
    img = io.imread(os.path.join(path, 'med filter', 'pruned.tif'))
    img_center = np.genfromtxt(os.path.join(path,'included_angle.csv'), delimiter=',', usecols=(3,4), skip_header=True)

    img_center_shift_x = int(img_center[0][1])
    img_center_shift_y = int(img_center[0][0])
    img_center_x = int(img.shape[1]/2)
    img_center_y = int(img.shape[0]/2)

    slice_range = np.min([img_center_y + img_center_shift_y, img_center_y - img_center_shift_y, img_center_x + img_center_shift_x, img_center_x - img_center_shift_x])

    img_invert = - img[
            img_center_y-img_center_shift_y-slice_range:img_center_y-img_center_shift_y+slice_range,
            img_center_x+img_center_shift_x-slice_range:img_center_x+img_center_shift_x+slice_range
            ]

    img_invert -= np.min(img_invert)
    return img_invert

def fill_holes(binary_img):
    """
    fill holes inside the binary image

    Parameters
    ----------
    binary_img : array_like

    Returns
    -------
    binary_img_filled : array_like
    """

    corner_filled_0 = flood_fill(binary_img, (0,0), 1, footprint=disk(1))
    corner_filled_1 = flood_fill(binary_img, (0,binary_img.shape[1]-1), 1, footprint=disk(1))
    corner_filled_2 = flood_fill(binary_img, (binary_img.shape[0]-1,0), 1, footprint=disk(1))
    corner_filled_3 = flood_fill(binary_img, (binary_img.shape[0]-1,binary_img.shape[1]-1), 1, footprint=disk(1))
    binary_holes = ~(corner_filled_0+corner_filled_1+corner_filled_2+corner_filled_3).astype(bool)
    binary_img_filled = np.logical_or(binary_img, binary_holes)
    return binary_img_filled

sample_list = np.sort([s for s in os.listdir(folder) if not s.startswith('.') and os.path.isdir(os.path.join(folder, s))])
sample_path_list = [os.path.join(folder, s) for s in sample_list]
img_list = []
labels_list = []

for sample in sample_list:
    sample_path = os.path.join(folder, sample)
    img_invert = img_slice(sample_path)

    mask_inner = np.zeros_like(img_invert).astype(bool)
    mask_inner[int(mask_inner.shape[0]/2-10):int(mask_inner.shape[0]/2+10), int(mask_inner.shape[1]/2-10):int(mask_inner.shape[1]/2+10)] = 1

    th_h = np.zeros_like(img_invert)
    th_h[:,:] = np.max(img_invert)
    th_h[mask_inner] = th_h_ratio*np.max(img_invert[mask_inner])
    th_l = th_l_ratio*np.median(img_invert[~mask_inner])

    labels = apply_hysteresis_threshold(img_invert, th_l, th_h)
    labels = fill_holes(labels)

    img_list.append(-img_invert)
    labels_list.append(labels)

print(sample_list[i])

viewer = napari.Viewer()
img_layer = viewer.add_image(img_list[i])

if save:
    label_layer = viewer.add_labels(labels_list[i], name='label to save')
    napari.run()
    output_path = os.path.join(sample_path_list[i], 'center radius')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_layer.save(os.path.join(output_path, 'radius.tif'))
    label_layer.save(os.path.join(output_path, 'radius_label.tif'))
    print('\n==========saved!==========')
else:
    label_layer = viewer.add_labels(labels_list[i], name="===don't save===")
    napari.run()
