## Description ##

This is a workflow to analyze VIPP1 polymerization on support lipid bilayers (SLB).

## Requirement ##

1. operating system for code development: macOS Sonoma Version 14.0
2. imageJ
    - install: `brew install --cask imagej`
3. python3
    - packages: `pip install -r requirements.txt`

## Content ##

### SVD flatten ###

**processing in imageJ:**

HS-AFM data saved as `tiff` stacks were preliminarily flattened and aligned using the in-lab imageJ plugin `BIOAFMLAB_HSAFM Movie Flattener` and `BIOAFMLAB_HSAFM Movie Aligner`.

**processing in python:**

1. run `0-mask_generator.py` to generate the mask of input movies.
2. run `1.1-line_align.py` and `1.2-protein_align.py` sequentially.
3. run `2-SVD_flatten.py` to get flattened output movies.

### Polygon analysis ###

**processing in imageJ:**

The mask of striations on VIPP1 polygons is generated in imageJ by median filter z-projection as `example data/median.tif`. The Niblack local thresholding was used to generate striation masks. The noise points in the mask were removed manually.

**processing in python:**

1. run `1-hough_transform.py` to get the numbers and included angles of striations.
    - parameters:
        - `img_center`: the coordinates of polygon's center
        - `rotate_angle`: the angle to rotate the image to put all striations in four quadrants
        - `center_mask_r`: the radius of mask in the center
        - `num_peaks`: the list of striation numbers in four quadrants

2. run `2-center_radius.py` to get the area of polygon's center.

### Growth analysis ###

**processing in imageJ**

First, all frames of HS-AFM were labeled with a sequential number, which served as a timestamp. After noisy frames were removed, frame labels were exported as `timestamp.csv`.

**processing in python**

1. run `1-label.py` to generate `roi_label.tif`, from which the label index of an isolated and unperturbed growing assembly can be found.
2. run `2-get_isolated_assemblies.py` to get growth data including morphologies, area, perimeters and growth boundary distances.
    - parameters: 
        - `roi_label_list`: the list of label indexes of the isolated assembly read from the file `roi_label.tif`
        - `beg`: the frame number when the isolated assembly appears in the movie
        - `end`: the frame number when the isolated assembly becomes unqualified
        - `save`: the list of properties of the assembly to save.

### Angle-distance simulation ###

run the file `sim_angle_distance.py`.

### Monte Carlo simulation ###

The parameters for simulation is in the file `params.py`.

1. To generate simulated VIPP1 polymerization trajectories, run the file `simulate_multiprocess.py`
2. To analyze simulated trajectories, run the file `analysis_ensemble.py`
