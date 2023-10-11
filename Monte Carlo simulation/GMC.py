import numpy as np
import pandas as pd
import params as p
import geometry as g

def get_E_init(traj):
    count = traj['count'].values
    curvature_avg = traj['curvature_avg'].values
    x_inner = traj['x_inner'].values
    y_inner = traj['y_inner'].values
    x_outer = traj['x_outer'].values
    y_outer = traj['y_outer'].values
    curvature_opt = g.get_intrinsic_curvature_from_count(count)
    rigidity = g.get_rigidity_from_count(count)
    energy_opt = g.get_intrinsic_energy_from_count(count)

    E_init = g.get_sum_energy(count, curvature_avg, rigidity, curvature_opt, energy_opt, x_inner, y_inner, x_outer, y_outer)
    return E_init

def get_ron_lateral(traj, r_on=p.r_on):
    cvt = traj['curvature_avg'].values[0]
    cnt = traj['count'].values[0]
    eng = g._get_internal_energy(1/cvt, cnt)

    ron_lateral = np.zeros(len(traj))
    ron_lateral[0] = r_on*np.exp(-eng)

    return ron_lateral

def get_roff_longitudinal(traj, strd, E_adh=p.E_adh, r_fil=p.r_fil):
    roff_longitudinal = np.zeros(len(strd))
    E_init = get_E_init(traj)

    flag = 0
    max_delta_E = 0
    crash_strand = None

    for i in range(len(strd)):
        count = traj['count'].values.copy()

        if len(count) == 1 and count[0] == 1:
            roff_longitudinal[i] = 0
        else:
            strand_end = strd.loc[i, 'end']
            count[strand_end-1] -= 1
            if not count[-1]:
                count = count[:-1]
            rigidity = g.get_rigidity_from_count(count)
            curvature_opt = g.get_intrinsic_curvature_from_count(count)
            energy_opt = g.get_intrinsic_energy_from_count(count)
            curvature_temp = g.get_empirical_curvature_from_count(count)
            curvature_avg = g.average_curvature(curvature_temp, rigidity)
            angle_avg = g.get_angle_from_curvature(curvature_avg)
            x_inner, y_inner, x_outer, y_outer = g.get_boundary_coord(angle_avg, count)

            E_minus = g.get_sum_energy(count, curvature_avg, rigidity, curvature_opt, energy_opt, x_inner, y_inner, x_outer, y_outer)

            np.seterr(over='raise')
            try:
                roff_longitudinal[i] = r_fil*np.exp(-E_adh+E_init-E_minus)
            except FloatingPointError:
                flag = 1
                if E_init - E_minus > max_delta_E:
                    crash_strand = i
                continue

    return roff_longitudinal, flag, crash_strand

def generate_random_pair(traj, strd, r_on=p.r_on, E_adh=p.E_adh, r_fil=p.r_fil):
    aoff, flag, crash_strand = get_roff_longitudinal(traj, strd)

    if flag:
        tau = 0
        index = crash_strand
    else:
        aon = get_ron_lateral(traj)
        afil = r_fil*np.ones_like(aoff)

        a = np.concatenate([aoff, afil, aon]) # filament shorten; filament lengthen; filament thicken

        tau_rng = np.random.default_rng()
        tau = tau_rng.exponential(scale=1/np.sum(a))

        mu_rng = np.random.default_rng()
        mu = np.sum(a)*mu_rng.random()
        index = np.where(np.cumsum(a)>=mu)[0][0]

    return tau, index

def update_population(traj, strd, tau, index):
    strd_num = len(strd)
    count = traj['count'].values.copy()

    if index < strd_num:
        count[strd.loc[index, 'end']-1] -= 1
        strd.loc[index, 'end'] -= 1
        if not count[-1]:
            count = count[:-1]
        if strd.loc[index, 'end'] == strd.loc[index, 'beg']:
            strd = strd.drop(index)

    elif index < 2*strd_num:
        if strd.loc[index-strd_num, 'end'] == len(count):
            count = np.concatenate([count, [1]])
        else:
            count[strd.loc[index-strd_num, 'end']] += 1
        strd.loc[index-strd_num, 'end'] += 1

    else:
        count[index-2*strd_num] += 1
        strd.loc[strd_num] = [strd['step'].unique()[0], index-2*strd_num, index-2*strd_num+1]

    curvature_opt = g.get_intrinsic_curvature_from_count(count)
    rigidity = g.get_rigidity_from_count(count)
    curvature_temp = g.get_empirical_curvature_from_count(count)
    curvature_avg = g.average_curvature(curvature_temp, rigidity)
    angle_avg = g.get_angle_from_curvature(curvature_avg)
    x_center, y_center = g.get_center_coord(angle_avg)
    x_inner, y_inner, x_outer, y_outer = g.get_boundary_coord(angle_avg, count)

    time = traj['time'].values[-1] + tau
    step = traj['step'].values[-1] + 1
    traj_updated = pd.DataFrame({
        'step':step,
        'time':time,
        'site':np.arange(len(count)),
        'count':count,
        'curvature_avg':curvature_avg,
        'x_center':x_center,
        'y_center':y_center,
        'x_inner':x_inner,
        'y_inner':y_inner,
        'x_outer':x_outer,
        'y_outer':y_outer,
        })

    strd['step'] = step
    strd = strd.reset_index(drop=True)

    return traj_updated, strd

if __name__ == '__main__':
    pass
