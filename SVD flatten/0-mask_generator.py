from skimage import io
import os
pwd = os.path.dirname(__file__)
from skimage.filters import threshold_otsu

folder = pwd
sample_path = 'example data'

img_stack = io.imread(os.path.join(folder, sample_path, 'input.tif'))

th = threshold_otsu(img_stack)
mask = (img_stack > th)
io.imsave(os.path.join(folder, sample_path, '0-mask.tif'), mask.astype('uint'))
