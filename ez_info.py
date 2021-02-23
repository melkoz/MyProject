import numpy as np
import scipy
import MDAnalysis as mda

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from org_functions import labels_path, saving_data, read_data, Timer, wrap


def distance(a, b,box_dim):
    ''' distance between two particles in a periodic system'''
    Lx = box_dim[0]
    Ly = box_dim[1]
    Lz = box_dim[2]
    
    dx = abs(a[0] - b[0])
    x_coor = min(dx, abs(Lx - dx))
     
    dy = abs(a[1] - b[1])
    y_coor = min(dy, abs(Ly - dy))
     
    dz = abs(a[2] - b[2])
    z_coor = min(dz, abs(Lz - dz))
 
    return np.sqrt(x_coor**2 + y_coor**2 + z_coor**2)

def eigen_zundel(u_conf,st=0,fin=None,every=1, dist_acc_crit = 1.30):
#     dist_acc_crit is in Angstrom
    fin = fin or len(u_conf.trajectory)
    
    oxygens_sel = (u_conf.atoms.names == 'O')
    hydrogens_sel = (u_conf.atoms.names == 'H')
    ox_nr = len(u_conf.trajectory[0].positions[oxygens_sel,:])
    hyd_nr = len(u_conf.trajectory[0].positions[hydrogens_sel,:])
    
    oxy_id = np.where(oxygens_sel)[0]
    hyd_id = np.where(hydrogens_sel)[0]
    
    # declare arrayays to calculate quantities
    dyd_per_oxy = np.zeros( [len(u_conf.trajectory[st:fin:every]),ox_nr])
    z_dim_oxy = np.zeros([len(u_conf.trajectory[st:fin:every]),ox_nr])
    z_dim_hyd = np.zeros(len(u_conf.trajectory[st:fin:every]))
    
    ei_zu_sel_bool = np.zeros([len(u_conf.trajectory[st:fin:every]),ox_nr],dtype=bool)
    
    eigen_z_dim = np.zeros([len(u_conf.trajectory[st:fin:every]),ox_nr])
    zundel_z_dim = np.zeros([len(u_conf.trajectory[st:fin:every]),ox_nr])
    # stores id of oxygens that form a hydronium
    # stores id of hydrogens that form a hydronium
    OH_dis_molec = np.zeros([len(u_conf.trajectory[st:fin:every]),ox_nr,3,3])
    hydr_info = np.zeros([len(u_conf.trajectory[st:fin:every]),4,3,3])
    zundel_info = np.zeros([len(u_conf.trajectory[st:fin:every]),2])
    eigen_info = np.zeros(len(u_conf.trajectory[st:fin:every]))
    
    frames = np.zeros(len(u_conf.trajectory[st:fin:every]))
    eigen_zundel_norm = np.zeros(2)
    
    print(len(frames))
    chain_count = 0
    chain_mo_count = 0
    eigen_count = 0 # one oxygene
    zundel_count = 0 # two oxygenes participate
    
    for t,ts in enumerate(u_conf.trajectory[st:fin:every]):
        oxy_pos = ts.positions[oxygens_sel,:]
        hyd_pos = ts.positions[hydrogens_sel,:]
        frames[t] = ts.frame
        print('<<<<',ts.frame,'>>>>')
        r=0
        for i in range(0,ox_nr):
            
            aa=0
            for j in range(0,hyd_nr):
                dist_to_H = distance(oxy_pos[i],hyd_pos[j], u_conf.dimensions[:3])
                if dist_to_H < dist_acc_crit:
                    dyd_per_oxy[t,i] += 1
                    OH_dis_molec[t,i,aa,0]= i  # collecting oxygen ind
                    OH_dis_molec[t,i,aa,1]= j  # collecting hydrogen ind
                    OH_dis_molec[t,i,aa,2]= dist_to_H # collecting the O-H distances 
                    aa +=1
                    
            if (dyd_per_oxy[t,i] > 2):
                hydr_info[t,r,:,:] = OH_dis_molec[t,i,:,:]
      # storing indexys of oxygens and hydrogens and distance between them with more than 3 hydrogens to access id: oxy_id[i]

                r+=1
                z_dim_oxy[t,i] = oxy_pos[i,2]
                ei_zu_sel_bool[t,i] = True 
#                 eigen_z_dim[t,i] = oxy_pos[i,2]
                
                eigen_count += 1
        if (dyd_per_oxy[t,:] > 2).sum() ==1:
            eig_ox = np.int(hydr_info[t,0,0,0])
            eigen_info[t] = oxy_id[eig_ox]
            
        if (dyd_per_oxy[t,:] > 2).sum() ==2:  
        # if two oxygens are detected with 3 hydrogens then it is associated with zundel structure
            
            eigen_count -= 2
            
            ox = 0
            ox1 = np.int(hydr_info[t,ox,0,0])
            ox2 = np.int(hydr_info[t,ox+1,0,0])
            zund_hydrogen = np.intersect1d(hydr_info[t,ox,:,1], hydr_info[t,ox+1,:,1])
            
            if (len(zund_hydrogen) == 1):
                zundel_count +=1
                zundel_info[t,0] = oxy_id[ox1]
                zundel_info[t,1] = oxy_id[ox2]

                
                ei_zu_sel_bool[t,ox1] = False 
                ei_zu_sel_bool[t,ox2] = False
                oo_dist = distance(oxy_pos[ox1],     
                                   oxy_pos[ox2], u_conf.dimensions[:3]) # distance between zundel oxygens

                z_dim_hyd[t] = hyd_pos[np.int(zund_hydrogen),2]
            
            if (len(zund_hydrogen) == 0):
                print('Chain    !!!!!    <<<<         <<<<<<<<<<        !!!!!!!!!!!      ')
                chain_count+=1
                ei_zu_sel_bool[t,ox1] = False 
                ei_zu_sel_bool[t,ox2] = False
                
        if ((dyd_per_oxy[t,:] > 2).sum() >2): 
            eigen_count -= (dyd_per_oxy[t,:] > 2).sum()
            print('Chain    !!!!!    <<<<         <<<<<<<<<<        !!!!!!!!!!!      ',(dyd_per_oxy[t,:] > 2).sum())
            for nr in range(0,(dyd_per_oxy[t,:] > 2).sum()):
                oxy = np.int(hydr_info[t,nr,0,0])
                print(oxy_id[oxy])
#                 z_dim_oxy[t,oxy] = 0 ????? zero this chain ones
                chain_mo_count+=1
                ei_zu_sel_bool[t,oxy] = False
                 
    print('eigen_count',eigen_count)
    print('zundel_count',zundel_count)
    print('chain_count',chain_count)
    print('chain_mo_count', chain_mo_count)
    eigen_zundel_norm[0] = eigen_count 
    eigen_zundel_norm[1] = zundel_count
    eigen_zundel_norm = eigen_zundel_norm/eigen_zundel_norm.sum()
   
    # PLOTTING
  #  
  #  tit_y = 'Time [ps]'
  #  tit_x = 'z direction [$\AA$]'
  #  fs =18
  #  plt.figure(figsize = (8,6))
  #  t_st = 0.0004 #ps

  #  for a in range(0,ox_nr):
  #      if z_dim_oxy[:,a].any() > 0.0:
  #          y = frames.copy()*t_st
  #          x = z_dim_oxy[:,a].copy()
  #          x[x==0]= np.nan
  #          plt.plot(x, y,'o',label=oxy_id[a])

  #  plt.xlabel(tit_x,fontsize=fs)
  #  plt.ylabel(tit_y,fontsize=fs)
  #  plt.tick_params(axis='x', labelsize=fs)
  #  plt.tick_params(axis='y', labelsize=fs)
  #  plt.legend(loc=2, prop={'size': 18})
  #  plt.tight_layout()
  #  plt.show()

    
    return  frames,z_dim_oxy,oxy_id, ei_zu_sel_bool, eigen_zundel_norm, z_dim_hyd, zundel_info, eigen_info

