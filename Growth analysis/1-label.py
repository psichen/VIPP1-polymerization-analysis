import napari
import numpy as np
import os
pwd = os.path.dirname(__file__)
import re
from skimage import io
from skimage.measure import label

folder = pwd

img = io.imread(os.path.join(folder, 'example data', 'input.tif'))
roi_label_file = os.path.join(folder, 'example data', 'roi_label.tif')

# reading existing `roi_label` & `label_selected`
#----------------------------------------
if os.path.exists(roi_label_file):
    roi_label = io.imread(roi_label_file)
    if os.path.exists(os.path.join(folder, 'example data', 'output')):
        labels_selected = np.zeros_like(roi_label, dtype='int')
        subset_list = [d for d in os.listdir(os.path.join(folder, 'example data', 'output')) if not d.startswith('.')]
        patch_count = 0
        for subset in subset_list:
            m = re.search(r'(\d+)_(\d+)', subset)
            beg = int(m.group(1))
            end = int(m.group(2))
            growth_list = [f for f in os.listdir(os.path.join(folder, 'example data', 'output', subset)) if f.startswith('growth')]
            roi_label_subset_path = os.path.join(folder, 'example data', 'output', subset, 'roi_label_subset.tif')
            roi_label_subset = io.imread(roi_label_subset_path)
            for growth in growth_list:

                # about labeling
                m = re.search(r'_(\d+)', growth)
                single_label = int(m.group(1))

                roi_label_temp = roi_label_subset.copy()
                roi_label_temp[roi_label_subset!=single_label] = 0

                # check duplication
                if np.sum(labels_selected[beg:end+1][roi_label_temp.astype(bool)]):
                    print('\nduplicated selected patches!')

                labels_selected[beg:end+1] += roi_label_temp
                patch_count += 1
#========================================

# generating `roi_label`
#----------------------------------------
else:
    mask_list = np.zeros((0, img.shape[1], img.shape[2])).astype(bool)
    from skimage.filters import threshold_otsu
    for frame in np.arange(len(img)):
        th = threshold_otsu(img[frame])
        mask = (img[frame] > th)
        mask_list = np.concatenate((mask_list, [mask]), axis=0)

    roi_label = np.empty((0, img.shape[1], img.shape[2])).astype(int)
    for i in np.arange(mask_list.shape[0])+1:
        temp_label = label(mask_list[:i,:,:], connectivity=3)
        roi_label = np.concatenate((roi_label, [temp_label[-1,:,:]]), axis=0)

    io.imsave(os.path.join(folder, 'example data', 'roi_label.tif'), roi_label)
#========================================

viewer = napari.Viewer()
img_layer = viewer.add_image(img)
viewer.add_labels(roi_label, opacity=.1)
if 'labels_selected' in locals():
    viewer.add_labels(labels_selected, opacity=1, blending='additive')
napari.run()
if 'subset_list' in locals():
    print(f'\ntotal labeled patches: {patch_count}')
