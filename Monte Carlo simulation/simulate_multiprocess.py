import os
pwd = os.path.dirname(__file__)
import numpy as np
import pandas as pd
import params as p
import geometry as g
import GMC as gmc
import multiprocessing

run_list = 100
cpu_workers = 4

def simulate(run):
    folder = pwd
    output_path = os.path.join(folder, "output", str(run))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        with open(os.path.join(folder, 'output/params.txt'), 'w') as f:
            f.write(f"r = {p.r}\n")
            f.write(f"l = {p.l}\n")
            f.write(f"w = {p.w}\n")
            f.write(f"rg = {p.rg}\n")
            f.write(f"e = {p.e}\n")
            f.write(f"r_on = {p.r_on}\n")
            f.write(f"r_fil = {p.r_fil}\n")
            f.write(f"E_adh = {p.E_adh}\n")
            f.write(f"step = {p.step}\n")

    if os.path.exists(os.path.join(output_path, 'traj.h5')):
        traj_prev = pd.read_hdf(os.path.join(output_path, 'traj.h5'), key='traj')
        traj_prev = traj_prev.loc[traj_prev['step']==traj_prev['step'].unique()[-1]]
        strd_prev = pd.read_hdf(os.path.join(output_path, 'strd.h5'), key='strands')
        strd_prev = strd_prev.loc[strd_prev['step']==strd_prev['step'].unique()[-1]]

    else:
        traj_prev = pd.DataFrame({
        'step':[0],
        'time':[0],
        'site':[0],
        'count':[1],
        'curvature_avg':[p.cur_list[0]],
        'x_center':g.get_center_coord(np.array([p.a]))[0],
        'y_center':g.get_center_coord(np.array([p.a]))[1],
        'x_inner':g.get_boundary_coord(np.array([p.a]), np.array([1]))[0],
        'y_inner':g.get_boundary_coord(np.array([p.a]), np.array([1]))[1],
        'x_outer':g.get_boundary_coord(np.array([p.a]), np.array([1]))[2],
        'y_outer':g.get_boundary_coord(np.array([p.a]), np.array([1]))[3],
        })

        strd_prev = pd.DataFrame({
            'step':[0],
            'beg':[0],
            'end':[1],
            })

    i = 0
    while True:
        tau, ind = gmc.generate_random_pair(traj_prev, strd_prev)
        traj_next, strd_next = gmc.update_population(traj_prev, strd_prev, tau, ind)
        if tau>=1e-6:
            try:
                traj_next.to_hdf(os.path.join(output_path, 'traj.h5'), key='traj', mode='a', append=True, index=False, complevel=1)
                strd_next.to_hdf(os.path.join(output_path, 'strd.h5'), key='strands', mode='a', append=True, index=False, complevel=1)
            except:
                import time
                time.sleep(.5)
                traj_next.to_hdf(os.path.join(output_path, 'traj.h5'), key='traj', mode='a', append=True, index=False, complevel=1)
                strd_next.to_hdf(os.path.join(output_path, 'strd.h5'), key='strands', mode='a', append=True, index=False, complevel=1)
        else:
            pass

        i += 1
        if i<p.step:
            traj_prev = traj_next
            strd_prev = strd_next
        else:
            break

if __name__ == "__main__":
    pool = multiprocessing.Pool(cpu_workers)
    pool.map(simulate, range(run_list))
    pool.close()
