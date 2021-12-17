import numpy as np
import sys
import os
import xpcs
import time

# variables for log-filenames
date = time.strftime("%Y-%m-%d")
timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
random_id = np.random.randint(0, 999)
job_name = f"{timestamp}-{random_id:03}"
job_file = f"{'./jobs'}/{job_name}.job"

print(f"Generating and submitting sbatch for job {job_name}")

datadir = './tmp/'+date+'/' # define path for saving data

# create a directory for the log-files
if not os.path.exists(datadir):
    os.mkdir(datadir)

################# Define parameters ##################
print('TTC analysis:')

# define run parameters
sample = '100NP_water_2'
scans = range(252,352)    # scans to run -- NP in water in focus
#scans = range(7587,7597)    # scans to run -- NP in water/DMSO in focus
#scans = range(7607,7617)    # scans to run -- NP in water/DMSO 1000um out
#scans = range(7637,7667)    # scans to run -- NP in water/DMSO 2000um out
exp_time = 1e-5           # exposure time [s]
downsample = 10            # pixel binning
n_ttc = 1000               # no of ttc:s per run
n_frames = 100            # no of frames per ttc
center_x, center_y = 1069.5, 201.5 # tristan detector center
print(sample, scans, exp_time, 'downsample:', downsample, 'n_ttc:', n_ttc)

# define rois
rois = [(8+i*1, 8+(i+1)*1) for i in range(4)] # 10x10 binning

####################################################

# experimental parameters
distance = 4.0 # m
photon_energy = 10.0e3 #eV
h = 4.135667516e-15
c = 2.99792458e8
wavelength = h*c / photon_energy
rs = [0.5*(roi[0]+roi[1])*downsample*55e-6 for roi in rois]
tth = np.arctan2(rs, distance)
q = 4.0e-10 * np.pi / wavelength * np.sin(0.5*tth)
print(q)

##################################################

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

roi_img = xpcs.make_roi(center_x, center_y, rois, downsample) # make downsampled roi image

def worker(arg, roi_img, exp_time, downsample):
    i, scan = arg
    frames = xpcs.make_frames('/data/visitors/nanomax/20210623/2021052608/raw/sample/tristan-%04d*' %scan, 
                         exp_time, roi_img, downsample)
    ttc0 = []
    for k in range(n_ttc):
        ttc0.append(xpcs.calc_ttc(frames[0+k*n_frames:n_frames*(k+1)], roi_img))
        
    return np.array(ttc0) 

pool = Pool(32)
ttc = pool.map(partial(worker, roi_img=roi_img, exp_time=exp_time, downsample=downsample), enumerate(scans))

# replace nan with zeros
ttc = np.array(ttc)

ttc_av = np.nanmean(ttc, axis=0) # averaging over runs
ttc_av = np.nanmean(ttc_av,axis=0) # averaging over runs
ttc_av0 = np.nanmean(ttc,axis=1) # average within each run

ttc_all = np.vstack(ttc) # stack consecutive runs
ttc_av1000 = np.nanmean(ttc_all[::1000], axis=0) # average every 1000th TTC 
ttc_av100 = np.nanmean(ttc_all[::100], axis=0) # average every 100th TTC 
ttc_av10 = np.nanmean(ttc_all[::10], axis=0) # average every 10th TTC 

ttc_av00 = ttc_all[0] # first TTC
id_nan = np.isnan(ttc_av00)
ttc_av00[id_nan] = 0

print('Analysis completed')

##################################################

# save data
np.save(datadir+sample+'_ttc_av_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', ttc_av)
np.save(datadir+sample+'_ttc_av0_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', ttc_av0)
np.save(datadir+sample+'_ttc_av1000_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', ttc_av1000)
np.save(datadir+sample+'_ttc_av100_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', ttc_av100)
np.save(datadir+sample+'_ttc_av10_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', ttc_av10)
np.save(datadir+sample+'_ttc_av00_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', ttc_av00)
np.save(datadir+sample+'_q_ttc_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', q)

print('Data saved')
