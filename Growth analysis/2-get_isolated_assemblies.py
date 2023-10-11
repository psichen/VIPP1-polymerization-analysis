import sys
import numpy as np
import os
pwd = os.path.dirname(__file__)
from matplotlib import pyplot as plt
from skimage import io
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
from skimage.transform import warp_polar

# initializing & reading
#----------------------------------------
folder = pwd
date = 'example data'
# chirality:    0-achiral: most of achiral objects are polygonal patches
#               1-right-handed: thumb towards the center of the spiral and other four fingers towards the filament growth direction
#               2-left-handed
# rotation:     0-static
#               1-clockwise
#               2-counter-clockwise
# morphology:   0-noncanonical: spiral first, then polygonal
#               1-spiral
#               2-polygonal
#               3-undiscernible
# class:        0-isolated, no newly emergent patches along the edge, start from zero
#               1-isolated, no newly emergent patches along the edge, pre-existing
#               2-isolated, has newly emergent patches along the edge

# e.g. save = [1,1,1,0] for right-handed, clockwise rotating, nascent spirals
save = [] #chirality, rotation, morphology, class.
roi_label_list = [1]
beg, end = 0,50
edit = 0
plot = 1

folder = os.path.join(folder, date)
img_path = os.path.join(folder, 'input.tif')
timestamp_path = os.path.join(folder, 'timestamp.csv')

img = io.imread(img_path)
timestamp = np.genfromtxt(timestamp_path, dtype='int')

img = img[beg:end+1]
timestamp = timestamp[beg:end+1]

if os.path.exists(os.path.join(folder, f'output/{beg}_{end}/roi_label_subset.tif')):
    labels = io.imread(os.path.join(folder, f'output/{beg}_{end}/roi_label_subset.tif'))
else:
    mask_path = os.path.join(folder, 'roi_label.tif')
    mask = io.imread(mask_path).astype(bool)
    mask = mask[beg:end+1]
    labels = label(mask, connectivity=3)

labels_selected = np.zeros_like(labels, dtype='int') # label layer for selected region in `roi_label_list`
labels_centroid = np.zeros_like(labels, dtype='int') # label layer with centroid coordinates of selected region
growth_data = pd.DataFrame()
#========================================
 
# get properties of individual label
#----------------------------------------
if edit:
    roi_label_list = []
for roi_label in np.sort(roi_label_list):
    labels_single = np.zeros_like(labels) # label layer for the individual assembly
    labels_selected[labels==roi_label] = roi_label
    labels_single[labels==roi_label] = roi_label

    b0,b1,b2,e0,e1,e2 = regionprops(labels_single)[0]['bbox'] # bbox indexes of the individual patch
    roi_binary = regionprops(labels_single)[0]['image'] # local binary mask of the individual patch

    # averaged centroid coordinates of the first 5 frames
    centroid = np.zeros(2)
    centroid_global = np.zeros(2)
    for i in np.arange(5):
        centroid += regionprops(roi_binary[i].astype(int))[0]['centroid']
        centroid_global += regionprops(labels_single[b0+i])[0]['centroid']
    centroid /= 5
    centroid_global /= 5

    centroid_global = np.rint(centroid_global).astype(int)
    labels_centroid[b0:, centroid_global[0], centroid_global[1]] = roi_label+1

    # get properties vs. timestamp
    for i in np.arange(labels_single.shape[0]):
        props = regionprops_table(labels_single[i], properties=['label', 'area', 'perimeter'])
        growth_data_per_frame = pd.DataFrame(props)

        growth_data_per_frame['key'] = f'{date}_{roi_label_list[0]}_{beg}_{end}'
        growth_data_per_frame['timestamp'] = timestamp[i]
        growth_data_per_frame['frame'] = i

        if i >= b0:
            roi_i = i - b0
            polar = warp_polar(roi_binary[roi_i], center=centroid)
            dist = np.max(np.cumsum(polar, axis=1), axis=1)
        else:
            dist = np.repeat(np.nan, 360)

        polar_data = pd.DataFrame([dist], columns=['dist-'+str(i) for i in np.arange(360)])
        growth_data_per_frame = pd.concat([growth_data_per_frame, polar_data], axis=1)
        growth_data = pd.concat([growth_data, growth_data_per_frame], ignore_index=True)

growth_data = growth_data.dropna()
#========================================

# plotting
#----------------------------------------
if plot:
    import napari
    viewer = napari.Viewer()
    img_layer = viewer.add_image(img)
    if len(roi_label_list):
        label_layer_centroid = viewer.add_labels(labels_centroid)
        label_layer = viewer.add_labels(labels_selected)
    else:
        label_layer = viewer.add_labels(labels)

if len(roi_label_list):
    f1, ax1 = plt.subplots(1,2, figsize=(12,6))
    f2, ax2 = plt.subplots(1,2, figsize=(12,6))
    for key, grp in growth_data.groupby('label'):
        ax1[0].plot(grp['frame'], grp['perimeter'], 'o', label=key)
        ax2[0].plot(grp['frame'], grp['area'], 'o', label=key)
        ax1[1].plot(grp['timestamp'], grp['perimeter'], 'o', label=key)
        ax2[1].plot(grp['timestamp'], grp['area'], 'o', label=key)

    ax1[0].set_title('perimeter')
    ax1[1].set_title('perimeter')
    ax1[0].set_xlabel('frame')
    ax1[1].set_xlabel('timestamp')
    ax1[0].set_ylabel('unscaled perimeter')
    ax1[1].set_ylabel('unscaled perimeter')
    ax1[0].legend()
    ax1[1].legend()

    ax2[0].set_title('area')
    ax2[1].set_title('area')
    ax2[0].set_xlabel('frame')
    ax2[1].set_xlabel('timestamp')
    ax2[0].set_ylabel('unscaled area')
    ax2[1].set_ylabel('unscaled area')
    ax2[0].legend()
    ax2[1].legend()

    for roi_label in roi_label_list:
        data = growth_data[growth_data['label'] == roi_label]
        timestamp_duration = data['timestamp'].values
        timestamp_duration -= timestamp_duration[0]
        dist_data = data[['dist-'+str(i) for i in np.arange(360)]].values

        f3= plt.figure()
        plt.imshow(dist_data[:,::-1])
        plt.xlabel('angle / degree')
        plt.ylabel('frame')
        plt.title(f'roi_label: {roi_label}')

        f4 = plt.figure()
        plt.axes(projection = 'polar')
        plt.ylim([0,1.1*np.max(dist_data)])
        plt.title(f'roi_label: {roi_label}')

        # plot leading edge every 20 timestamps
        while timestamp_duration[-1] > 20:
            j = np.where(timestamp_duration > 20)[0][0]
            plt.polar(2*np.pi - np.arange(360)*np.pi/180, dist_data[j,:])
            timestamp_duration -= timestamp_duration[j]

    if plot:
        plt.show()

if plot:
    napari.run()
#========================================

# saving
#----------------------------------------
if edit and not len(roi_label_list):
    if not os.path.exists(os.path.join(folder, f'output/{beg}_{end}')):
        os.makedirs(os.path.join(folder, f'output/{beg}_{end}'))
    label_layer.save(os.path.join(folder, f'output/{beg}_{end}/roi_label_subset.tif'))
    print('save roi_label_subset!')

if len(save) == 4 and len(roi_label_list):
    if not os.path.exists(os.path.join(folder, f'output/{beg}_{end}')):
        os.makedirs(os.path.join(folder, f'output/{beg}_{end}'))

    if not os.path.exists(os.path.join(folder, f'output/{beg}_{end}/roi_label_subset.tif')):
        io.imsave(os.path.join(folder, f'output/{beg}_{end}/roi_label_subset.tif'), labels, check_contrast=False)

    chirality  = ['achiral','right-handed','left-handed']
    rotation   = ['static','clockwise','counter-clockwise']
    morphology = ['noncanonical','spiral','polygonal','undiscernible']

    growth_output = os.path.join(folder, f'output/{beg}_{end}/growth_{roi_label_list[0]}.csv')
    img_subset_output = os.path.join(folder, f'output/{beg}_{end}/img_subset_{roi_label_list[0]}.tif')
    peri_output   = os.path.join(folder, f'output/{beg}_{end}/peri_{roi_label_list[0]}.png')
    area_output   = os.path.join(folder, f'output/{beg}_{end}/area_{roi_label_list[0]}.png')
    polar_output  = os.path.join(folder, f'output/{beg}_{end}/polar_{roi_label_list[0]}.png')
    aniso_output  = os.path.join(folder, f'output/{beg}_{end}/aniso_{roi_label_list[0]}.png')

    growth_data['chirality']  = chirality[save[0]]
    growth_data['rotation']   = rotation[save[1]]
    growth_data['morphology'] = morphology[save[2]]
    growth_data['class'] = save[3]

    growth_data.to_csv(growth_output, index=False)
    io.imsave(img_subset_output, img[b0:e0,b1:e1,b2:e2])
    f1.savefig(peri_output)
    f2.savefig(area_output)
    f3.savefig(polar_output)
    f4.savefig(aniso_output)
    print('save data!')
#========================================
