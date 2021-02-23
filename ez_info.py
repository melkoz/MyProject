import numpy as np
import MDAnalysis as mda

import matplotlib.pyplot as plt
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
    
    distance = np.sqrt(x_coor**2 + y_coor**2 + z_coor**2)
    
    return distance


def distance_arr(a, bb,box_dim):
    ''' distance between two particles in a periodic system'''
    Lx = box_dim[0]
    Ly = box_dim[1]
    Lz = box_dim[2]
    b = np.array(bb.copy())
    
    dx = np.abs(a[0] - b[:][:,0])
    x_coor = np.minimum(dx[:], abs(Lx - dx[:]))
    
    dy = abs(a[1] - b[:][:,1])
    y_coor = np.minimum(dy[:], abs(Ly - dy[:]))
    
    dz = abs(a[2] - b[:][:,2])
    z_coor = np.minimum(dz[:], abs(Lz - dz[:]))
 
    return np.sqrt(x_coor**2 + y_coor**2 + z_coor**2)

def eigen_zundel(u_conf,st=0,fin=None,every=1, dist_acc_crit = 1.30):
#     dist_acc_crit is in Angstrom
    fin = fin or len(u_conf.trajectory)
    
    
    oxygens_sel = (u_conf.atoms.names == 'O')
    hydrogens_sel = (u_conf.atoms.names == 'H')
    ion_sel = (u_conf.atoms.names != 'C') & (u_conf.atoms.names != 'O') & (u_conf.atoms.names != 'H')
    
    ox_nr = len(u_conf.trajectory[0].positions[oxygens_sel,:])
    hyd_nr = len(u_conf.trajectory[0].positions[hydrogens_sel,:])
    
    oxy_id = np.where(oxygens_sel)[0]
    hyd_id = np.where(hydrogens_sel)[0]
    
    # declare arrayays to calculate quantities

    xyz_hyd = np.zeros([len(u_conf.trajectory[st:fin:every]),3])
    xyz_ion = np.zeros([len(u_conf.trajectory[st:fin:every]),3])
    dist_ion_hyd = np.zeros(len(u_conf.trajectory[st:fin:every]))
    dist_ion_hyd_min = np.zeros(len(u_conf.trajectory[st:fin:every]))
    
    # stores id of oxygens and hydrogens that form a hydronium
    hydr_info = np.zeros([4,3,2])
   
    
    zundel_info = np.zeros([len(u_conf.trajectory[st:fin:every]),2])
    eigen_info = np.zeros(len(u_conf.trajectory[st:fin:every]))
    
    frames = np.zeros(len(u_conf.trajectory[st:fin:every]))
    eigen_zundel_norm = np.zeros(2)
    
    print(len(frames))
    
    eigen_count = 0 # one oxygen
    zundel_count = 0 # two oxygenes participate
    
    for t,ts in enumerate(u_conf.trajectory[st:fin:every]):
        oxy_pos = ts.positions[oxygens_sel,:]
        hyd_pos = ts.positions[hydrogens_sel,:]
        xyz_ion[t,:] = ts.positions[ion_sel,:]
        
        frames[t] = ts.frame
        print('<<<<',ts.frame,'>>>>')
        
        inter_ox =0
        for i in range(0,ox_nr):
            # calculate distances between i-th oxygen and all hydrogens
            O_Hs_distances = distance_arr(oxy_pos[i],hyd_pos, u_conf.dimensions[:3] )
            
            # array with indexes of hydrogens bonded to i-th oxygen
            OH_bonded = np.argwhere(O_Hs_distances < dist_acc_crit)
            
            if len(OH_bonded) > 2:
                for g in range(len(OH_bonded)):    
                    hydr_info[inter_ox,g,0] = i                # index of the oxygen in eigen/zundel
                    hydr_info[inter_ox,g,1] = OH_bonded[g][0]  # index of the hydrogens in eigen/zundel
                
                inter_ox +=1
                eigen_count += 1
                
                
        if inter_ox == 1:
        # ifonly one oxygen is detected with 3 hydrogens then it is associated with eigen structure    
            eig_ox = np.int(hydr_info[0,0,0])
            
            eigen_info[t] = oxy_id[eig_ox]   # translation from index in oxy_pos to the oxygen_id in MD
            xyz_hyd[t,:] = oxy_pos[eig_ox,:]
            
        if inter_ox == 2:  
        # if two oxygens are detected with 3 hydrogens then it is associated with zundel structure
            eigen_count -= 2
            
            ox = 0
            ox1 = np.int(hydr_info[ox,0,0])
            ox2 = np.int(hydr_info[ox+1,0,0])
            zund_hydrogen = np.intersect1d(hydr_info[ox,:,1], hydr_info[ox+1,:,1])
            
            if (len(zund_hydrogen) == 1):
                zundel_count +=1
                zundel_info[t,0] = oxy_id[ox1]   # translation from index in oxy_pos to the oxygen_id in MD
                zundel_info[t,1] = oxy_id[ox2]   # translation from index in oxy_pos to the oxygen_id in MD

                xyz_hyd[t,:] = hyd_pos[np.int(zund_hydrogen),:]
                
        if (inter_ox >2): 
            eigen_count -= inter_ox
        
          
        dist_ion_hyd[t] = distance(xyz_hyd[t,:], xyz_ion[t,:], u_conf.dimensions[:3])
        dist_ion_hyd_min[t] = min(distance_arr(xyz_ion[t,:],hyd_pos, u_conf.dimensions[:3] ))
    
    eigen_zundel_norm[0] = eigen_count 
    eigen_zundel_norm[1] = zundel_count
    eigen_zundel_norm = eigen_zundel_norm/eigen_zundel_norm.sum()
   
    return  frames, eigen_zundel_norm, dist_ion_hyd, dist_ion_hyd_min, xyz_ion, xyz_hyd, zundel_info, eigen_info


