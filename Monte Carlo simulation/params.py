import numpy as np
import os
pwd = os.path.dirname(__file__)

r = 13 # intrinsical radius (nm)
l = 4.8 # subunit length (nm)
a = l/r # intrinsical bond angle
w = 5 # subunit width (nm)
rg = .1 # bending rigidity (nm*kBT)

e = 2 # Lennard Jones interaction strength (kBT)

r_on = .5 # lateral association rate (s-1)
r_fil = 6 # longitudinal association rate (s-1)
E_adh = 3 # subunit adhesion energy (kBT)

step = 12000

cur_list_path = os.path.join(pwd, 'intrinsic_curvature.npy')
eng_list_path = os.path.join(pwd, 'intrinsic_energy.npy')
if os.path.exists(cur_list_path) and os.path.exists(eng_list_path):
    cur_list = np.load(cur_list_path)
    eng_list = np.load(eng_list_path)
else:
    cur_list = None
    eng_list = None

if __name__ == "__main__":
    pass
