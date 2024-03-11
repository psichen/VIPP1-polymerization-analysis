from matplotlib.dviread import enum
import params as p
import numpy as np
from scipy.optimize import minimize
import os

# vector calculation
#----------------------------------------
def magnitude(vec):
    if len(vec.shape) == 1:
        return np.sqrt(np.sum(vec**2))
    else:
        return np.sqrt(np.sum(vec**2, axis=len(vec.shape)-1))

def normalize_vec(vec):
    if len(vec.shape) == 1:
        return vec/magnitude(vec)
    elif len(vec.shape) == 2:
        l = magnitude(vec)
        l = np.repeat(l, 2).reshape((-1,2))
        return vec/l

def rotate_vec(vec, delta):
    theta = np.arctan(vec[1]/vec[0])
    if vec[0] > 0 and vec[1] > 0:
        pass
    elif vec[0] < 0 and vec[1] > 0:
        theta += np.pi
    elif vec[0] < 0 and vec[1] < 0:
        theta += np.pi
    elif vec[0] > 0 and vec[1] < 0:
        theta += 2*np.pi
    elif vec[0] == 0 and vec[1] > 0:
        pass
    elif vec[0] == 0 and vec[1] < 0:
        theta += 2*np.pi

    theta += delta
    return magnitude(vec)*np.array([np.cos(theta), np.sin(theta)])

#========================================

# curvature calculation from energy minimization
#----------------------------------------
def _get_radius(r, count, width=p.w):
    """get radius array from intrinsical radius and count

    :r: intrinsical radius of filament with [count] subunits (nm)
    :count: subunit count
    :width: subunit width (nm)
    :returns: radius array (nm)

    """
    n = np.arange(count).astype(float)
    return r + (n - (count-1)/2)*width

def _get_internal_energy(r, count, length=p.l, width=p.w, r0=p.r, rigidity=p.rg):
    """get internal energy including bending and stretching from intrinsical radius and count

    :r: intrinsical radius of filament with [count] subunits (nm)
    :count: subunit count
    :length: subunit length (nm)
    :width: subunit width (nm)
    :r0: intrinsical radius of single strand filament (nm)
    :rigidity: bending rigidity of single strand filament (nm*kBT)
    :returns: bending energy (kBT)

    """
    r_list = _get_radius(r, count, width)
    l_list = length/r*r_list

    # neglect stretch deformation due to sliding of subunits
    return np.sum(.5*rigidity*(1/r_list - 1/r0)**2*l_list)

#========================================

# spiral calculation
#----------------------------------------
def get_angle_from_curvature(curvature, length=p.l):
    """get angle from curvature

    :curvature: curvature array
    :length: subunit length (nm)
    :returns: angle array

    """
    return length*curvature

def get_rigidity_from_count(count, width=p.w, rigidity=p.rg):
    """get the rigidity from thickness according to elasticity theory

    :count: count array
    :width: width of subunit (nm)
    :rigidity: bending rigidity of single strand (nm*kBT)
    :returns: rigidity array (nm*kBT)

    """
    rgdt = rigidity*count**3
    return rgdt

def get_intrinsic_curvature_from_count(count, curvature_list=p.cur_list):
    """get intrinsic curvature from count

    :count: count array
    :curvature_list: optimized curvature array (nm-1)
    :returns: curvature array (nm-1)

    """
    c = curvature_list[count-1]

    return c

def get_empirical_curvature_from_count(count, width=p.w):
    return 1/(width*count*4.4+27)

def get_intrinsic_energy_from_count(count, energy_list=p.eng_list):
    """get intrinsic energy from count

    :count: count array
    :energy_list: optimized energy array (kBT)
    :returns: energy array (kBT)

    """
    E = energy_list[count-1]

    return E

def average_curvature(curvature, rigidity, window=7):
    """rigidity-weighted moving average curvature

    :curvature: curvature array (nm-1)
    :rigidity: rigidity array (kBT)
    :window: moving window
    :returns: curvature array (nm-1)

    """
    offset = (window-1)//2
    cvt = np.concatenate([np.zeros(offset), curvature, np.zeros(offset)])
    rgd = np.concatenate([np.zeros(offset), rigidity, np.zeros(offset)])
    s_cvt = np.cumsum(np.insert(cvt*rgd,0,0))
    s_rgd = np.cumsum(np.insert(rgd,0,0))
    wma = (s_cvt[window:]-s_cvt[:-window])/(s_rgd[window:]-s_rgd[:-window])
    return wma

#========================================

# coordinates calculation
#----------------------------------------
def get_center_coord(angle, length=p.l):
    """get center coordinates from bond angle and subunit length

    :angle: angle array
    :length: subunit length (nm)
    :returns: x_coord array; y_coord array (nm)

    """
    radius = length/angle
    cum_ang = -np.cumsum(angle)
    cos = np.cos(cum_ang)
    sin = np.sin(cum_ang)
    ag_x = np.repeat(-1, len(angle))
    ag_y = np.repeat(0, len(angle))

    rg_x = (ag_x*cos - ag_y*sin)
    rg_y = (ag_x*sin + ag_y*cos)

    rg_x = np.concatenate([[-1], rg_x])
    rg_y = np.concatenate([[0], rg_y])

    side_x = radius*np.diff(rg_x)
    side_y = radius*np.diff(rg_y)

    co_x = np.cumsum(side_x) - radius[0]
    co_y = np.cumsum(side_y)

    return co_x, co_y

def get_boundary_coord(angle, count, length=p.l, width=p.w):
    """get boundary coordinates from bond angle and subunit count

    :angle: angle array
    :count: count array
    :length: subunit length (nm)
    :width: subunit width (nm)
    :returns: x_coord array; y_coord array (nm)

    """
    x_center, y_center = get_center_coord(angle, length)

    x_aug = -1*magnitude(np.array([x_center[0], y_center[0]]))
    y_aug = 0

    x_center_aug = np.concatenate([[x_aug], x_center])
    y_center_aug = np.concatenate([[y_aug], y_center])

    x_vec = np.diff(x_center_aug)
    y_vec = np.diff(y_center_aug)
    thick_vec = np.concatenate([
        -y_vec.reshape((len(y_vec),-1)),
        x_vec.reshape((len(x_vec),-1))
        ], axis=1)
    count_multipler = np.repeat(count.reshape((len(count),1)), 2, axis=1)
    thick_vec = count_multipler*width*normalize_vec(thick_vec)

    x_inner = x_center - .5*width*normalize_vec(thick_vec)[:,0]
    y_inner = y_center - .5*width*normalize_vec(thick_vec)[:,1]
    x_outer = x_inner + thick_vec[:,0]
    y_outer = y_inner + thick_vec[:,1]

    return x_inner, y_inner, x_outer, y_outer

def get_dist_tensor(x_inner, y_inner, x_outer, y_outer, count, width=p.w):
    """get all coordinates between inner and outer boundaries

    :x_inner; y_inner: inner coordinates array (nm)
    :x_outer; y_outer: outer coordinates array (nm)
    :count: count array
    :width: subunit width (nm)
    :returns: x_coord array; y_coord array (nm)

    """
    site_num = len(count)
    dist_tensor = np.empty((site_num, site_num))

    co_inner_boundary = np.concatenate([x_inner.reshape((site_num, -1)), y_inner.reshape((site_num, -1))], axis=1)
    co_outer_boundary = np.concatenate([x_outer.reshape((site_num, -1)), y_outer.reshape((site_num, -1))], axis=1)
    vec_boundary = co_outer_boundary - co_inner_boundary

    co_outer_bead = co_outer_boundary - normalize_vec(vec_boundary)*.5*width
    co_inner_bead = co_inner_boundary + normalize_vec(vec_boundary)*.5*width

    vec_bead = co_outer_bead - co_inner_bead

    tensor_outer_bead = np.repeat(co_outer_bead, site_num, axis=0).reshape((site_num, site_num, 2))
    tensor_side1 = tensor_outer_bead - co_inner_bead
    tensor_side2 = tensor_outer_bead - co_outer_bead

    mag_side1 = magnitude(tensor_side1)
    mag_side2 = magnitude(tensor_side2)
    side = np.minimum(mag_side1, mag_side2)

    dot1 = np.sum(tensor_side1*vec_bead, axis=2)
    dot2 = np.sum(tensor_side2*vec_bead, axis=2)
    dot = np.sign(dot1*dot2)
    cross = np.cross(tensor_side1, vec_boundary)/magnitude(vec_boundary)

    dist_tensor[dot < 0] = cross[dot < 0]
    dist_tensor[dot >= 0] = side[dot >= 0]

    return np.abs(dist_tensor)

#========================================

# energy calculation
#----------------------------------------
def LJ_potential(d, sigma, epsilon):
    return 4*epsilon*((sigma/d)**12 - (sigma/d)**6)
    
def get_LJP_from_coord(x_inner, y_inner, x_outer, y_outer, count, sigma=p.w, epsilon=p.e):
    """Lennard Jones potential (exclusion energy) calculation
    :x_inner; y_inner: inner coordinates array (nm)
    :x_outer; y_outer: outer coordinates array (nm)
    :count: count array
    :sigma: distance where the potential is zero (nm)
    :epsilon: interaction strength (kBT)
    :returns: LJ potential (kBT)

    """
    if len(count) == 1:
        return 0
    else:
        dist = get_dist_tensor(x_inner, y_inner, x_outer, y_outer, count, width=sigma)
        dist = dist[~np.eye(len(dist)).astype(bool).reshape(len(dist), -1)]

        if np.sum(dist<1.12*sigma):
            np.seterr(divide='raise')
            try:
                LJP = np.sum(LJ_potential(dist[dist<1.12*sigma], sigma, epsilon))
            except FloatingPointError:
                LJP = np.inf
        else:
            LJP = 0

        return LJP

def get_bending_energy(curvature, rigidity, intrinsic_curvature, intrinsic_energy, length=p.l):
    """get bending energy of the entire filament

    :curvature: average curvature array (nm-1)
    :rigidity: rigidity array (nm*kBT)
    :intrinsic_curvature: optimized curvature from energy minimization (nm-1)
    :intrinsic_energy: optimized energy from energy minimization (kBT)
    :length: subunit length (nm)
    :returns: bending energy array

    """
    BE = intrinsic_energy + .5*rigidity*(curvature-intrinsic_curvature)**2*length
    return BE

def get_sum_energy(count, curvature, rigidity, curvature_opt, energy_opt, x_inner, y_inner, x_outer, y_outer):
    """get sum of energy including bending energy and Lennard-Jones energy 

    :cont: subunit count
    :curvature: averaged curvature array (nm-1)
    :rigidity: bending rigidity array (nm*kBT)
    :curvature_opt: optimized curvature array (nm-1)
    :energy_opt: optimized energy array (kBT)
    :x_inner; y_inner: inner coordinates array (nm)
    :x_outer; y_outer: outer coordinates array (nm)
    :returns: energy sum (kBT)

    """
    bending_energy = get_bending_energy(curvature, rigidity, curvature_opt, energy_opt)
    exclusion_energy = get_LJP_from_coord(x_inner, y_inner, x_outer, y_outer, count)
    E = np.sum(bending_energy) + np.sum(exclusion_energy)

    return E

#========================================

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    precalculation = 0
    internal_energy = 0
    delta_energy = 0
    min_energy = 1

    if internal_energy:
        def show_internal_energy(rds, cnt):
            eng = np.zeros_like(rds)
            for i,r in enumerate(rds):
                eng[i] = _get_internal_energy(r, cnt)

            plt.figure()
            plt.plot(rds, eng)
            plt.xlabel('radius (nm)')
            plt.ylabel('energy (kBT)')

            plt.figure()
            plt.plot(rds, np.exp(-eng))
            plt.xlabel('radius (nm)')
            plt.ylabel('rate ratio')
            plt.show()
            return eng

        rds = np.arange(30,300).astype(float)
        eng = show_internal_energy(rds, 1)
        average_ratio = np.sum(np.exp(-eng[rds>=40])*rds[rds>=40])/np.sum(rds[rds>=40])
        print(f"average ratio: {average_ratio}")

    if delta_energy:
        cnt = np.arange(300)+1
        cvt_emp = get_empirical_curvature_from_count(cnt)
        cvt_opt = get_intrinsic_curvature_from_count(cnt)
        rgd = get_rigidity_from_count(cnt)
        eng = get_intrinsic_energy_from_count(cnt)
        eng_bend = get_bending_energy(cvt_emp, rgd, cvt_opt, eng)

        plt.plot(cnt[1:], np.diff(eng_bend))
        plt.show()

    if min_energy:
        def show_internal_energy(rds, cnt):
            eng = np.zeros_like(rds)
            for i, r in enumerate(rds):
                eng[i] = _get_internal_energy(r, cnt)
            return eng

        cnt = 50
        col_r = (235-161)/(cnt-1)*np.arange(cnt)+161
        col_g = (139-199)/(cnt-1)*np.arange(cnt)+199
        col_b = (140-231)/(cnt-1)*np.arange(cnt)+231
        col_r /= 255
        col_g /= 255
        col_b /= 255

        x = []
        y = []
        for i in np.arange(cnt):
            rds = np.linspace(10,300,5000).astype(float)
            rds = rds[rds>i/2*p.w]
            eng = show_internal_energy(rds, i+1)
            x.append(rds[eng==np.min(eng)])
            y.append(eng[eng==np.min(eng)])
            plt.plot(rds, eng, color=(col_r[i], col_g[i], col_b[i]))

        plt.plot(x,y, color='k')
        plt.xlim(10,200)
        plt.ylim(-1,30)
        plt.xlabel('radius of curvature')
        plt.ylabel('bending energy')
        plt.show()

    if precalculation:
        cnt = np.arange(900)+1
        r = []
        E = []
        for n in cnt:
            if n == 1:
                r_init = p.r
            else:
                r_init = res.x[0]
            res = minimize(_get_internal_energy, r_init, args=(n, p.l, p.w, p.r, p.rg), tol=1e-4)
            if not res.success:
                print(f'subunit {n}: failed!')
            r.append(res.x[0])
            E.append(res.fun)
        E = np.array(E)
        r = np.array(r)
        c = 1/r

        np.save(os.path.join(os.path.dirname(__file__), 'example_data', 'intrinsic_curvature'), c)
        np.save(os.path.join(os.path.dirname(__file__), 'example_data', 'intrinsic_energy'), E)
