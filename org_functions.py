import numpy as np
import scipy
import scipy.stats as sps
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from scipy.stats import linregress
import MDAnalysis as mda
import time





def labels_path(all_pos):
    labs =[]
    for i in range(len(all_pos)):
        separator = '_'
        lab = separator.join(all_pos[i].split('/')[-1].split('.')[0].split('-')[0].split('_')[:-1])
        labs.append(lab)
    return labs

def saving_data(keys, values,path_name):
    ''' key = lables, names of data
    values= list with arrays to store
    name_path= "path/name" '''
    dicts = {}
    for i,ke in enumerate(keys):
            dicts[ke] = values[i]
    np.savez(path_name,**dicts)
    print("Data are saved in .npz in", path_name)
#     dic = np.load('dicts.npz')
#     print(dic.files, dic.files[0])
#     dic['g34'][:,0]#

def read_data(path_name):
    full_name = path_name + '.npz'
    myfile = np.load(full_name, allow_pickle=True)
    print('Keys in the file:  ',myfile.files)
    print('Access elements by: myfile[myfile.files[i]]')
    return myfile 


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = (time.perf_counter() - self._start_time)/60
        elapsed_time_s = (time.perf_counter() - self._start_time)
        self._start_time = None
        print(f"Elapsed time: {elapsed_time_s:0.4f} s which is {elapsed_time:0.4f} minutes")
        
        
def wrap(ts):
    '''Takes positions (u.trajectory[frame_number].positions) from a universe
    and translates atoms such that all fit in the box'''
    x,y,z = 0,1,2
    
    if (ts.dimensions[0] == 0):
        print("Dimensions of the box are probably wrong.",ts.dimensions)
    else:
        L_x = ts.dimensions[x]
        L_y = ts.dimensions[y]
        L_z = ts.dimensions[z]

        ts.positions[:,x] = ts.positions[:,x] - (ts.positions[:,x]//L_x)*L_x
        ts.positions[:,y] = ts.positions[:,y] - (ts.positions[:,y]//L_y)*L_y
        ts.positions[:,z] = ts.positions[:,z] - (ts.positions[:,z]//L_z)*L_z   
    return ts.copy()
