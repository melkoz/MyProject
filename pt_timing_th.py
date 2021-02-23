import numpy as np
import scipy
import MDAnalysis as mda

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from org_functions import labels_path, saving_data, read_data, Timer, wrap



def nr_fr_range(arr_ranges):
    nr_fr = arr_ranges[:,1] - arr_ranges[:,0]
    return nr_fr

def time_fr_to_fs(data_ez, nr_fr, ts_fs = 0.4):
    nr_fr_skip = np.abs(data_ez[0][0] - data_ez[0][1])
    
    time_fs = nr_fr * nr_fr_skip*ts_fs
    return time_fs

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    
    absdiff = np.abs(np.diff(iszero))
    
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def not_zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    
    absdiff = np.abs(np.diff(iszero))
    
    # Runs start and end where absdiff is 1.
    ranges_non_zero = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges_non_zero

def id_runs(a,id_at):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, id_at).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges



def plot_res_time(res_times,res_times2 ):    
    plt.figure(figsize=(6.26894*2,4.7747*1))
    fs = 18
    nr_bins = 20
    y_val = np.array(res_times.copy())/1000
    y_val_2 = np.array(res_times2.copy())/1000
    print('Average residence time with an oxygen', np.round(np.average(y_val),1))
    density, edges, patches = plt.hist(y_val,bins=nr_bins,
                                       histtype='step', label='Residence time with merging')#
    
    density2, edges2, patches2 = plt.hist(y_val_2, bins=edges, 
                                           histtype='step', label='Residence time no/less merging')#

    plt.tick_params(labelsize=fs)
    plt.xlabel("Time [ps]", fontsize=fs)
    plt.ylabel("Counts", fontsize=fs)
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)
    plt.legend(fontsize=fs-4)
    plt.show()

    
def plot_ox_id(ez_data, ox_id1,ox_id2, nr_ox_invol, t_st_fs):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6.26894*2,4.7747),sharey=True)
    fig.subplots_adjust(wspace=0)
    fs =18
    for i in range(0, nr_ox_invol):
        y = ox_id1[:,i].copy()
        y[y==0] = np.nan
        ax1.plot(ez_data[0].copy()*t_st_fs/1000,y, marker='o')
        
    for i in range(0, nr_ox_invol):
        y = ox_id2[:,i].copy()
        y[y==0] = np.nan
        ax2.plot(ez_data[0].copy()*t_st_fs/1000,y, marker='o')
        
    tit_x = 'Time [ps]'
    tit_y = 'Oxygen ID'
    ax1.set_ylabel(tit_y,fontsize=fs)
    for ax in (ax1,ax2):
        ax.set_xlabel(tit_x,fontsize=fs)
        
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)

    ax1.set_title('Without merging', fontsize=fs)
    ax2.set_title('With merging',fontsize=fs)
    plt.tight_layout() 
    plt.show()
    
    
def pt_rt(ez_data,nr_fr=1, merge_fr = 0,t_st_fs = 0.4, nr_tot_ox=128):
    less_merg = 5
    # selecting oxygen at id that are identified to interact with the excess proton
    appear_id = np.unique(np.concatenate([np.unique(ez_data[7].copy()[np.nonzero(ez_data[7].copy())]),
                                np.unique(ez_data[6][:,0].copy()[np.nonzero(ez_data[6][:,0].copy())]),
                               np.unique(ez_data[6][:,1].copy()[np.nonzero(ez_data[6][:,1].copy())])]))
    nr_fr_skip = np.abs(ez_data[0][0] - ez_data[0][1])
    

    nr_ox_invol = len(appear_id)
    resid_times = []
    resid_times_no_merg = []
    ox_id_timing = np.zeros([len(ez_data[0]),nr_ox_invol])
    ox_id_timing_cl = np.zeros([len(ez_data[0]),nr_ox_invol])
    for a in range(0, len(appear_id)):

        at_id = np.int(appear_id[a])
        eig_at = id_runs(ez_data[7],at_id)
        zund1_at = id_runs(ez_data[6][:,0],at_id)
        zund2_at = id_runs(ez_data[6][:,1],at_id)
        # combining ranges of frames from both eig-zund conformations in one array 
        # array that gives ranges when this ox at was interactring with H+
        all_ranges = np.concatenate([eig_at,zund1_at,zund2_at])

        # array with info, when ox interacting =1 if ox not interacting =0 
        #combined for both both eig-zund conformations
        ox_timing = np.zeros(len(ez_data[0]))
        ox_timing_no_merg = np.zeros(len(ez_data[0]))


        for p in all_ranges:
            for i in range(p[0],p[1],1):
                ox_timing[i] = 1
                ox_timing_no_merg[i] = 1
                ox_id_timing[i,a] = at_id
                ox_id_timing_cl[i,a] = at_id


        #filling gaps smaller than nr_zer with 1
        ox_zeros_ranges = zero_runs(ox_timing)
        for j in range(0,len(ox_zeros_ranges)):
            nr_zer = ox_zeros_ranges[j,1]-ox_zeros_ranges[j,0]
            if (nr_zer < merge_fr): 
                for i in range(ox_zeros_ranges[j][0],ox_zeros_ranges[j][1],1):
                    ox_timing[i] = 1.
                    ox_id_timing[i,a] = at_id            
        
        # remowing the les than merging times from the graph
        ox_non_zero_ranges = not_zero_runs(ox_id_timing[:,a])
        nr_non_zer = nr_fr_range(ox_non_zero_ranges)
        for g in range(0,len(ox_non_zero_ranges)):
            if (nr_non_zer[g] < merge_fr):
                for h in range(ox_non_zero_ranges[g][0], ox_non_zero_ranges[g][1]):
                    ox_id_timing[i,a] = 0
        
        
        # les merging runs
        with_at_rans = not_zero_runs(ox_timing)
        with_at_nr_fr = nr_fr_range(with_at_rans)
        ox_zeros_ranges2 = zero_runs(ox_timing_no_merg)
        for j in range(0,len(ox_zeros_ranges2)):
            nr_zer = ox_zeros_ranges2[j,1]-ox_zeros_ranges2[j,0]
            if (nr_zer < less_merg): 
                for i in range(ox_zeros_ranges2[j][0],ox_zeros_ranges2[j][1],1):
                    ox_timing_no_merg[i] = 1.
                    
        
        with_at_rans_no_merg = not_zero_runs(ox_timing_no_merg)
        with_at_nr_fr_no_merg = nr_fr_range(with_at_rans_no_merg)
        
        for k in range(0,len(with_at_nr_fr_no_merg)):
            resid_times_no_m = time_fr_to_fs(ez_data,with_at_nr_fr_no_merg[k])
            resid_times_no_merg.append(resid_times_no_m)
            
        # rejecting times that were shorter than merging criterium because 
        #this times were already colected by some other residence time
        with_at_sel = with_at_nr_fr[with_at_nr_fr>merge_fr]
        for i in range(0,len(with_at_sel)):
            res_t_ps = time_fr_to_fs(ez_data,with_at_sel[i])
            resid_times.append(res_t_ps)

    
 
    plot_ox_id(ez_data,ox_id_timing_cl,ox_id_timing, nr_ox_invol , t_st_fs)
#     plot_ox_id(ez_data,ox_id_timing, nr_ox_invol , t_st_fs)
    print('Time limit to merge data:',time_fr_to_fs(ez_data,merge_fr),'fs', 
          'and less merged data with limit: ', time_fr_to_fs(ez_data, less_merg),'fs')
    print('Minimum selested residence time:', min(resid_times),'fs')
    plot_res_time(resid_times, resid_times_no_merg)
    
    av_res_time = np.round(np.average(resid_times)/1000,1)
    sim_t= nr_fr*t_st_fs/1000
    
    hop_rate = np.round(len(resid_times)/sim_t,2)
    print('Rate of hops with merging:',hop_rate,'ps^(-1)')
    
    appear_id_aft_merg = np.unique(ox_id_timing[:,:].copy()[np.nonzero(ox_id_timing[:,:].copy())])
    print('Nr ox at detected before merging:',len(appear_id))
    print('Nr ox at detected after merging:',len(appear_id_aft_merg))
    participating_ox_perc = (len(appear_id_aft_merg)*100)/(nr_tot_ox*sim_t) # percent per ps
    return av_res_time, hop_rate,max(resid_times)/1000, participating_ox_perc,resid_times
