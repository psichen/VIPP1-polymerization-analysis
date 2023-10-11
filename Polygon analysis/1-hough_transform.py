import os
pwd = os.path.dirname(__file__)
import numpy as np
import csv
from skimage import io
from skimage.morphology import binary_closing
from skimage.morphology import binary_erosion
from skimage.morphology import diamond
from skimage.morphology import disk
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import pyplot as plt
from skimage.transform import rotate

def hough_extract(hspace, angles, distances, quadrant, th = .5, min_distance=100, min_angle=20, num_peaks=2):
    if quadrant == 'top-left':
        hspace_mask = np.ones_like(hspace)
        hspace_mask[:, :int(hspace.shape[1]/2)] = 0
        hspace_quadrant = np.ma.array(hspace, mask=hspace_mask, fill_value=0)
    elif quadrant == 'top-right':
        hspace_mask = np.ones_like(hspace)
        hspace_mask[int(hspace.shape[0]/2):, int(hspace.shape[1]/2):] = 0
        hspace_quadrant = np.ma.array(hspace, mask=hspace_mask, fill_value=0)
    elif quadrant == 'bottom-left':
        hspace_mask = np.ones_like(hspace)
        hspace_mask[int(hspace.shape[0]/2):, int(hspace.shape[1]/2):] = 0
        hspace_quadrant = np.ma.array(hspace, mask=hspace_mask, fill_value=0)
    elif quadrant == 'bottom-right':
        hspace_mask = np.ones_like(hspace)
        hspace_mask[int(hspace.shape[0]/4):-int(hspace.shape[0]/4), :int(hspace.shape[1]/2)] = 0
        hspace_quadrant = np.ma.array(hspace, mask=hspace_mask, fill_value=0)

    angle_list = np.array([])
    dist_list = np.array([])
    hspace_filled = hspace_quadrant.filled()
    for _, angle, dist in zip(*hough_line_peaks(hspace_filled, angles, distances, min_distance, min_angle, threshold = th * np.max(hspace), num_peaks=num_peaks)):
        angle_list = np.append(angle_list, angle)
        dist_list = np.append(dist_list, dist)

    return hspace_quadrant, angle_list, dist_list

def divided_hough(mask, img_center, center_mask_r=15, rotate_angle=0, th = .5, min_distance=100, min_angle=20, num_peaks=[2,2,2,2]):
    #center mask
    center_mask = disk(center_mask_r).astype(bool)
    pad_up = np.array((mask.shape[0]-center_mask.shape[0])/2).astype(int) - img_center[0]
    pad_down = mask.shape[0] - center_mask.shape[0] - pad_up
    pad_left = np.array((mask.shape[1]-center_mask.shape[1])/2).astype(int) + img_center[1]
    pad_right = mask.shape[1] -center_mask.shape[1] - pad_left
    center_mask = np.pad(center_mask, ((pad_up, pad_down), (pad_left, pad_right)), 'constant', constant_values=0)

    _img_center = np.empty_like(img_center)
    _img_center[0] = (np.array(mask.shape[0])/2).astype(int) - img_center[0]
    _img_center[1] = (np.array(mask.shape[1])/2).astype(int) + img_center[1]
    mask = rotate(mask, rotate_angle, center=(_img_center[1], _img_center[0]))

    f0, ax0 = plt.subplots(1,1)
    ax0.imshow(mask, extent=[-int(mask.shape[0]/2), mask.shape[0]-int(mask.shape[0]/2), -int(mask.shape[1]/2), mask.shape[1]-int(mask.shape[1]/2)], interpolation=None, cmap='gray')
    ax0.imshow(np.ma.array(center_mask, mask=~center_mask), extent=[-int(mask.shape[0]/2), mask.shape[0]-int(mask.shape[0]/2), -int(mask.shape[1]/2), mask.shape[1]-int(mask.shape[1]/2)], interpolation=None, alpha=.4, cmap='gray')
    ax0.axhline(-_img_center[0]+int(mask.shape[0]/2))
    ax0.axvline(_img_center[1]-int(mask.shape[1]/2))

    f1, ax1 = plt.subplots(2,2)
    f2, ax2 = plt.subplots(2,2)

    mask[center_mask] = 0
    included_angle = np.array([])
    quadrant_list = np.array([])
    subplot_list = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    for i in np.arange(4):
        if i == 0: #top-left
            subplot_mask = mask[:_img_center[0], :_img_center[1]]
            hspace, angles, distance = hough_line(subplot_mask)
            hspace_quadrant, angle_list, dist_list = hough_extract(hspace, angles, distance, subplot_list[i], th, min_distance, min_angle, num_peaks=num_peaks[i])
            included_angle = np.append(included_angle, np.pi/2-angle_list)
            quadrant_list = np.append(quadrant_list,np.repeat(subplot_list[i], len(angle_list)))

            ax1[int(i/2)][i%2].imshow(subplot_mask, interpolation=None, cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]], cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace_quadrant, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]])
            ax2[int(i/2)][i%2].set(xlabel='angles', ylabel='distance / pixel')

            for angle, dist in zip(angle_list, dist_list):
                (x,y) = dist*np.array([ np.cos(angle), np.sin(angle) ])
                ax1[int(i/2)][i%2].axline((x,y), slope=np.tan(angle+np.pi/2))

        elif i == 1: #top-right
            subplot_mask = mask[:_img_center[0], _img_center[1]:]
            hspace, angles, distance = hough_line(subplot_mask)
            hspace_quadrant, angle_list, dist_list = hough_extract(hspace, angles, distance, subplot_list[i], th, min_distance, min_angle, num_peaks=num_peaks[i])
            included_angle = np.append(included_angle, np.pi/2-angle_list)
            quadrant_list = np.append(quadrant_list,np.repeat(subplot_list[i], len(angle_list)))

            ax1[int(i/2)][i%2].imshow(subplot_mask, interpolation=None, cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]], cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace_quadrant, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]])
            ax2[int(i/2)][i%2].set(xlabel='angles', ylabel='distance / pixel')

            for angle, dist in zip(angle_list, dist_list):
                (x,y) = dist*np.array([ np.cos(angle), np.sin(angle) ])
                ax1[int(i/2)][i%2].axline((x,y), slope=np.tan(angle+np.pi/2))

        elif i == 2: #bottom-left
            subplot_mask = mask[_img_center[0]:, :_img_center[1]]
            hspace, angles, distance = hough_line(subplot_mask)
            hspace_quadrant, angle_list, dist_list = hough_extract(hspace, angles, distance, subplot_list[i], th, min_distance, min_angle, num_peaks=num_peaks[i])
            included_angle = np.append(included_angle, -np.pi/2-angle_list)
            quadrant_list = np.append(quadrant_list,np.repeat(subplot_list[i], len(angle_list)))

            ax1[int(i/2)][i%2].imshow(subplot_mask, interpolation=None, cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]], cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace_quadrant, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]])
            ax2[int(i/2)][i%2].set(xlabel='angles', ylabel='distance / pixel')

            for angle, dist in zip(angle_list, dist_list):
                (x,y) = dist*np.array([ np.cos(angle), np.sin(angle) ])
                ax1[int(i/2)][i%2].axline((x,y), slope=np.tan(angle+np.pi/2))

        elif i == 3: #bottom-right
            subplot_mask = mask[_img_center[0]:, _img_center[1]:]
            hspace, angles, distance = hough_line(subplot_mask)
            hspace_quadrant, angle_list, dist_list = hough_extract(hspace, angles, distance, subplot_list[i], th, min_distance, min_angle, num_peaks=num_peaks[i])
            included_angle = np.append(included_angle, -np.pi/2-angle_list)
            quadrant_list = np.append(quadrant_list,np.repeat(subplot_list[i], len(angle_list)))

            ax1[int(i/2)][i%2].imshow(subplot_mask, interpolation=None, cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]], cmap='gray')
            ax2[int(i/2)][i%2].imshow(hspace_quadrant, interpolation=None, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), distance[-1], distance[0]])
            ax2[int(i/2)][i%2].set(xlabel='angles', ylabel='distance / pixel')

            for angle, dist in zip(angle_list, dist_list):
                (x,y) = dist*np.array([ np.cos(angle), np.sin(angle) ])
                ax1[int(i/2)][i%2].axline((x,y), slope=np.tan(angle+np.pi/2))

    plt.tight_layout()
    plt.show()
    return included_angle, quadrant_list, f0, f1, f2

save = 0
dir_path = 'example data'
folder_dir = pwd
file_name = 'med filter/pruned_label.tif'
img_center = [0,0] # (y, x)
rotate_angle = 0
center_mask_r = 0
th = .3
min_distance = 20
min_angle = 20
num_peaks = [2,3,3,2] #top-left, top-right, bottom-left, bottom-right

file_path = os.path.join(folder_dir, dir_path, file_name)
mask = io.imread(file_path)

#connect line segments
footprint = diamond(3)
mask = binary_closing(mask, footprint)
#thin line segments
mask = binary_erosion(mask, footprint)

included_angle, quadrant_list, f0, f1, f2 = divided_hough(mask, img_center, center_mask_r, rotate_angle, th, min_distance, min_angle, num_peaks)
print(np.rad2deg(included_angle))

if save:
    header = ['file', 'quadrant', 'angle', 'image_center_y', 'image_center_x', 'rotate', 'center_mask_radius', 'hough_threshold', 'min_distance', 'min_angle', 'num_peaks_0', 'num_peaks_1', 'num_peaks_2', 'num_peaks_3']
    with open(os.path.join(folder_dir, dir_path, 'included_angle.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for angle, quadrant in zip(included_angle, quadrant_list):
            data = [dir_path, quadrant, np.rad2deg(angle), img_center[0], img_center[1], rotate_angle, center_mask_r, th, min_distance, min_angle, num_peaks[0], num_peaks[1], num_peaks[2], num_peaks[3]]
            writer.writerow(data)

    f0.savefig(os.path.join(folder_dir, dir_path, 'mask.png'))
    f1.savefig(os.path.join(folder_dir, dir_path, 'estimated_line.png'))
    f2.savefig(os.path.join(folder_dir, dir_path, 'hough_space.png'))
