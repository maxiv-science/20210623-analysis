import numpy as np
import sys
import os
import xpcs
import time

# variables for log-filenames
date = time.strftime("%Y-%m-%d")
timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")

print(f"Generating and submitting sbatch for job {timestamp}")

datadir = './tmp/'+date+'/' # define path for saving data

# create a directory for the job-files
if not os.path.exists(datadir):
    os.mkdir(datadir)

print('g2 analysis:')

################# Define parameters ##################

# define run parameters
sample = '100NP_water'
scans = range(252,552)    # scans to run -- NP in water in focus
#scans = range(7587,7597)    # scans to run -- NP in water/DMSO in focus
#scans = range(7607,7617)    # scans to run -- NP in water/DMSO 1000um out
#scans = range(7637,7667)    # scans to run -- NP in water/DMSO 2000um out
exp_time = 5e-7           # exposure time [s]
downsample = 10           # pixel binning
center_x, center_y = 1069.5, 201.5 # tristan detector center
print(sample, scans, exp_time, 'downsample:', downsample)

# define rois
nbins = 8
rois = [(8+i*1, 8+(i+1)*1) for i in range(nbins)] # 10x10 binning
print(rois)

# define time delays
dlin = np.arange(1, 100) # 100 first data points linspaced
dlog = np.logspace(2,5,300).astype(int) # rest logspaced
delays = np.concatenate((dlin, dlog), axis=None)

####################################################

# experimental parameters
distance = 4.0 # m
photon_energy = 10.0e3 #eV
h = 4.135667516e-15
c = 2.99792458e8
wavelength = h*c / photon_energy

# make the roi
roi_img = xpcs.make_roi(center_x, center_y, rois, downsample) # make downsampled roi image
print(roi_img.shape)

# make simple radial Q-map
x = np.arange(0,roi_img.shape[1])
y = np.arange(0,roi_img.shape[0])
xx,yy = np.meshgrid(x,y)
r = (((xx-center_x)**2 + (yy-center_y)**2 ) ** 0.5)
rs = r*pix
tth = np.arctan2(rs, distance)
Q = 4.0e-9 * np.pi / wavelength * np.sin(0.5*tth) # 1/nm
Q_downsample = xpcs.downsample_img(Q, downsample)/downsample**2
Q_map = np.zeros((515, 2069)) 
Q_map[0:515//downsample, 0:2069//downsample] = Q_downsample

q = np.empty((nbins,))
for i in range(0,nbins):
    inds = np.argwhere(roi_img.flatten()==i+1)
    q[i] = Q_map.flatten()[inds].mean()
print(q)

###################################################

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def worker(arg, roi_img, exp_time, downsample):
    i, scan = arg
    frames = xpcs.make_frames('/data/visitors/nanomax/20210623/2021052608/raw/sample/tristan-%04d*' %scan, 
                         exp_time, roi_img, downsample)
    return xpcs.calc_g2(frames, roi_img, delays)

pool = Pool(32)
g2 = pool.map(partial(worker, roi_img=roi_img, exp_time=exp_time, downsample=downsample), enumerate(scans))
g2_std = np.std(g2, axis=0)
g2_av = np.mean(g2, axis=0)

print('Analysis completed')

##################################################

# save data
np.save(datadir+sample+'_g2_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', np.array(g2))
np.save(datadir+sample+'_g2av_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', g2_av)
np.save(datadir+sample+'_g2std_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', g2_std)
np.save(datadir+sample+'_q_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', q)
np.save(datadir+sample+'_delays_'+str(scans)+'_'+str(downsample)+'x'+str(downsample)+'_'+str(exp_time)+'s', delays)
