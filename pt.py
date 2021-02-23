#!/usr/bin/env python

import MDAnalysis as mda
from ez_info import eigen_zundel
from org_functions import saving_data,read_data, wrap
from pt_timing_th import pt_rt

start_fr = 5000
box_S = [14.74, 12.76, 20.0 ,90.0, 90.0, 90.0 ]
univ_S = mda.Universe('./top.xyz','./pos.xyz', format="XYZ",dt=0.0004)
univ_S.dimensions = box_S
univ_S.trajectory.add_transformations(*[wrap])


print(univ_S.trajectory)

ez_S = eigen_zundel(univ_S,st=start_fr, every=100, dist_acc_crit = 1.32)

nr_frames = len(univ_S.trajectory[start_fr:])

res_time_S = pt_rt(ez_S, nr_frames , merge_fr=50, nr_tot_ox=34)
