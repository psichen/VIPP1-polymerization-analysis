import os
pwd = os.path.dirname(__file__)
import geometry as g
import params as p
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def set_time(traj, step=1):
    """set same time step for trajectories

    :traj: simulated trajectories
    :step: time step (s)
    :returns: trajectories with same time step

    """
    time_raw = traj['time'].unique()
    time = np.arange(0, time_raw[-1], step)
    traj_plot = pd.DataFrame()
    for i in range(len(time)):
        if i:
            j = np.where(time_raw<=time[i])[0][-1]
            traj_plot = pd.concat([traj_plot, traj.loc[traj['time']==time_raw[j]]], ignore_index=False)
    return traj_plot

def rotate_coord(x, y, theta):
    """rotate coordinates by multiplying rotatation matrix

    :x: x coordinates array
    :y: y coordinates array
    :theta: rotation angle
    :returns: x_rotated array; y_rotated array

    """
    x_rotated = x*np.cos(theta) - y*np.sin(theta)
    y_rotated = x*np.sin(theta) + y*np.cos(theta)
    return x_rotated, y_rotated

def anchor_coord(x_center, y_center, x_inner, y_inner, x_outer, y_outer, pos='tail'):
    """anchor one specific point of the spiral 

    :x_center: x center coordinates array at one time point
    :y_center: y center coordinates array at one time point
    :x_inner: x inner coordinates array at one time point
    :y_inner: y inner coordinates array at one time point
    :x_outer: x outer coordinates array at one time point
    :y_outer: y outer coordinates array at one time point
    :pos: anchored by 'head' or 'tail'
    :returns: x_anchored array; y_anchored array

    """
    x_cen_anchored = np.zeros_like(x_center)
    y_cen_anchored = np.zeros_like(y_center)
    x_in_anchored = np.zeros_like(x_inner)
    y_in_anchored = np.zeros_like(y_inner)
    x_out_anchored = np.zeros_like(x_outer)
    y_out_anchored = np.zeros_like(y_outer)

    if pos == 'tail':
        if len(x_center) < 2:
            vec = np.array([x_center[0], y_center[0]])
        else:
            vec = np.array([x_center[1]-x_center[0], y_center[1]-y_center[0]])
    elif pos == 'head':
        if len(x_center) < 2:
            vec = np.array([x_center[0], y_center[0]])
        else:
            vec = np.array([x_center[-2]-x_center[-1], y_center[-2]-y_center[-1]])

    nor = np.array([0, 1])
    theta = (np.sign(np.cross(nor, vec))) * np.arccos(np.sum(nor*vec)/np.sqrt(np.sum(vec**2)))

    x_cen_anchored, y_cen_anchored = rotate_coord(x_center, y_center, -theta)
    x_in_anchored, y_in_anchored = rotate_coord(x_inner, y_inner, -theta)
    x_out_anchored, y_out_anchored = rotate_coord(x_outer, y_outer, -theta)

    return x_cen_anchored, y_cen_anchored, x_in_anchored, y_in_anchored, x_out_anchored, y_out_anchored

def align_traj(traj, pos='tail'):
    """align coordinates of trajectories

    :traj: trajectories to align
    :pos: align by 'head' or 'tail'
    :returns: aligned trajectories

    """
    time = traj['time'].unique()
    traj_aligned = traj.copy()
    for t in time:
        tj = traj.loc[traj['time']==t]
        x_cen, y_cen, x_in, y_in, x_out, y_out = anchor_coord(tj['x_center'].values, tj['y_center'].values, tj['x_inner'].values, tj['y_inner'].values, tj['x_outer'].values, tj['y_outer'].values, pos)

        if pos == 'tail':
            x_shift = x_cen[0]
            y_shift = y_cen[0]
        elif pos == 'head':
            x_shift = x_cen[-1]
            y_shift = y_cen[-1]

        x_cen -= x_shift
        x_in -= x_shift
        x_out -= x_shift
        y_cen -= y_shift
        y_in -= y_shift
        y_out -= y_shift

        traj_aligned.loc[traj_aligned['time']==t, 'x_center'] = x_cen
        traj_aligned.loc[traj_aligned['time']==t, 'y_center'] = y_cen
        traj_aligned.loc[traj_aligned['time']==t, 'x_inner'] = x_in
        traj_aligned.loc[traj_aligned['time']==t, 'y_inner'] = y_in
        traj_aligned.loc[traj_aligned['time']==t, 'x_outer'] = x_out
        traj_aligned.loc[traj_aligned['time']==t, 'y_outer'] = y_out

    return traj_aligned

def get_origin(traj, length=p.l):
    x_h0 = traj['x_center'].values[-1]
    y_h0 = traj['y_center'].values[-1]
    x_h1 = traj['x_center'].values[-2]
    y_h1 = traj['y_center'].values[-2]
    c_0 = traj['curvature_avg'].values[-1]
    c_1 = traj['curvature_avg'].values[-2]

    theta = length*c_1
    side_vec = np.array([x_h1-x_h0, y_h1-y_h0])
    rds_vec = g.rotate_vec(side_vec, .5*(np.pi-theta))
    rds_vec = g.normalize_vec(rds_vec)/c_0
    x_ori = rds_vec[0] + x_h0
    y_ori = rds_vec[1] + y_h0

    return x_ori, y_ori

def show_angular_velocity(traj, show=0):
    x_ori, y_ori = get_origin(traj)
    site_num = traj['site'].max()

    t_list = np.empty(0).astype(float)
    r_list = np.empty(0).astype(float)
    w_list = np.empty(0).astype(float)

    for site in [0]:
        dp = traj.loc[traj['site']==site]
        time = dp['time'].unique()
        dx = np.diff(dp['x_center'].values)
        dy = np.diff(dp['y_center'].values)
        x = x_ori - dp['x_center'].values
        y = y_ori - dp['y_center'].values
        x = x[1:]
        y = y[1:]

        arc_vec = np.transpose(np.array([dx, dy]))
        rds_vec = np.transpose(np.array([x, y]))

        t = time[1:]
        r = np.sqrt(x**2 + y**2)
        w = np.sqrt(dx**2 + dy**2)/r/np.diff(time)*np.sign(np.cross(arc_vec, rds_vec))

        t_list = np.concatenate([t_list, t])
        r_list = np.concatenate([r_list, r])
        w_list = np.concatenate([w_list, w])

    if show:
        f = plt.figure()
        ax = f.add_subplot(projection='3d')
        ax.scatter(r_list, t_list, w_list, alpha=.7)
        ax.set_xlabel('radius')
        ax.set_ylabel('time')
        ax.set_zlabel('angular velocity')
        plt.show()

    return t_list, r_list, w_list

def show_spiral(traj, label=None, save=0):
    """show spiral growth animation w/ or w/o alignment

    :dp: trajectories dataframe to plot
    :returns: None

    """
    time = dp['time'].unique()

    x_max = np.max([dp['x_inner'].abs().max(), dp['x_outer'].abs().max()])
    y_max = np.max([dp['y_inner'].abs().max(), dp['y_outer'].abs().max()])

    f, ax = plt.subplots()
    l1, = ax.plot([], [])
    l2, = ax.plot([], [])
    l3, = ax.plot([], [], '*')
    t = ax.text(.9*x_max, .9*y_max, '', fontsize=15)

    def init():
        ax.set_xlim(-2*x_max, 2*x_max)
        ax.set_ylim(-2*y_max, 2*y_max)
        l1.set_data(dp.loc[dp['time']==time[0], 'x_inner'], dp.loc[dp['time']==time[0], 'y_inner'])
        l2.set_data(dp.loc[dp['time']==time[0], 'x_outer'], dp.loc[dp['time']==time[0], 'y_outer'])
        # l3.set_data(dp.loc[dp['time']==time[0], 'x_center'], dp.loc[dp['time']==time[0], 'y_center'])
        if label is not None:
            if len(dp.loc[dp['time']==time[0]]) >= label:
                l3.set_data(dp.loc[dp['time']==time[0], 'x_center'].values[label], dp.loc[dp['time']==time[0], 'y_center'].values[label])
            else:
                l3.set_data([], [])
        t.set_text(f'{np.around(time[0], 1)} s')
        # return l1, l2
        return l1, l2, l3

    def update(frame):
        l1.set_data(dp.loc[dp['time']==time[frame], 'x_inner'], dp.loc[dp['time']==time[frame], 'y_inner'])
        l2.set_data(dp.loc[dp['time']==time[frame], 'x_outer'], dp.loc[dp['time']==time[frame], 'y_outer'])
        # l3.set_data(dp.loc[dp['time']==time[frame], 'x_center'], dp.loc[dp['time']==time[frame], 'y_center'])
        if label is not None:
            if len(dp.loc[dp['time']==time[frame]]) >= label:
                l3.set_data(dp.loc[dp['time']==time[frame], 'x_center'].values[label], dp.loc[dp['time']==time[frame], 'y_center'].values[label])
            else:
                l3.set_data([], [])
        t.set_text(f'{np.around(time[frame], 1)} s')
        return l1, l2, l3

    ani = FuncAnimation(f, update, frames=np.arange(len(time)), init_func=init, interval=60)
    if save:
        ani.save('./spiral_growth.mp4')
    plt.show()

    return None

def show_area(dp, length=p.l, width=p.w):
    """show area of the polymer

    :dp: trajectories dataframe to plot
    :returns: None

    """
    time = dp['time'].unique()
    count = np.zeros_like(time)
    for i in range(len(time)):
        count[i] = dp.loc[dp['time']==time[i], 'count'].sum()
    area = count*length*width
    rate = np.diff(area)/np.diff(time)*6e-5 # um^2/min

    hist, bins = np.histogram(rate, bins=15)
    bins = .5*(bins[1:] + bins[:-1])
    mod = GaussianModel()
    par = mod.make_params(amplitude=5, center=.5, sigma=1)
    out = mod.fit(hist, par, x=bins)
    x_eval = np.linspace(np.min(bins), np.max(bins), 100)

    f, ax = plt.subplots(1, 2)
    ax[0].plot(time/60, area/1e6, '.-')
    ax[0].set_xlabel('time (min)')
    ax[0].set_ylabel('area (um^2)')

    ax[1].bar(bins, hist, color='white', edgecolor='k', width=np.mean(np.diff(bins)))
    ax[1].plot(x_eval, out.eval(x=x_eval))
    ax[1].set_xlabel('area rate (um^2/min)')
    ax[1].set_ylabel('count')
    plt.show()

    print(f"area rate: {out.params['center'].value} (um^2/min)")

    return None

def show_energy(dp):
    """show total energy (adhesion, bending, LJP) of the polymer

    :dp: trajectories dataframe to plot
    :returns: None

    """
    time = dp['time'].unique()
    energy = np.zeros_like(time)
    for i in range(len(time)):
        df = dp.loc[dp['time']==time[i]]
        count = df['count'].values
        curvature_avg = df['curvature_avg'].values
        curvature_opt = g.get_intrinsic_curvature_from_count(count)
        energy_opt = g.get_intrinsic_energy_from_count(count)
        rigidity = g.get_rigidity_from_count(count)
        x_inner = df['x_inner'].values
        y_inner = df['y_inner'].values
        x_outer = df['x_outer'].values
        y_outer = df['y_outer'].values
        E_adh = count*p.E_adh

        E_diss = g.get_sum_energy(count, curvature_avg, rigidity, curvature_opt, energy_opt, x_inner, y_inner, x_outer, y_outer)

        energy[i] = np.sum(E_diss) - np.sum(E_adh)

    time = time[energy <= 0]
    energy = energy[energy <= 0]
    plt.figure()
    plt.plot(time/60, energy, '.-')
    plt.xlabel('time (min)')
    plt.ylabel('energy (kBT)')
    plt.show()

F = os.path.join(pwd, 'output')
R = np.array([])
T = np.array([])
V = np.array([])

folder = F

if os.path.exists(os.path.join(folder, 'radius_list.npy')):
    radius_list = np.load(os.path.join(folder, 'radius_list.npy'))
    time_list = np.load(os.path.join(folder, 'time_list.npy'))
    velocity_list = np.load(os.path.join(folder, 'velocity_list.npy'))
else:
    time_list = np.empty(0).astype(float)
    radius_list = np.empty(0).astype(float)
    velocity_list = np.empty(0).astype(float)

    for i in range(100):
        traj = pd.read_hdf(os.path.join(folder, str(i), 'traj.h5'), key='traj')
        dp = set_time(traj, step=1)
        dp = align_traj(dp, pos='head')
        t,r,w = show_angular_velocity(dp, show=0)

        time_list = np.concatenate([time_list, t])
        radius_list = np.concatenate([radius_list, r])
        velocity_list = np.concatenate([velocity_list, w])

    np.save(os.path.join(folder, 'radius_list.npy'), radius_list)
    np.save(os.path.join(folder, 'time_list.npy'), time_list)
    np.save(os.path.join(folder, 'velocity_list.npy'), velocity_list)

R = radius_list
T = time_list
V = velocity_list

R = R[V>-1.5]
T = T[V>-1.5]
V = V[V>-1.5]

T = T[R>41]
V = V[R>41]
R = R[R>41]

f = plt.figure()
ax = f.add_subplot(projection='3d')
ax.scatter(R, T, V, marker=',', c=V, cmap='coolwarm', alpha=.3)
ax.set_xlabel('radius')
ax.set_ylabel('time')
ax.set_zlabel('angular velocity')

f = plt.figure()
plt.scatter(R, T, marker=',', c=V, cmap='coolwarm', alpha=.3)
plt.xlabel('radius')
plt.ylabel('time')
plt.colorbar()
# f.savefig(os.path.join(folder, 'angular_velocity.pdf'), transparent=True)

plt.show()
