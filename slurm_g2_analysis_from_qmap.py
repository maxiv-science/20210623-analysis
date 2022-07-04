import numpy as np
import sys
import os
import xpcs
import functions
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
sample = '100NP_dmso'       # sample name
#scans = range(252,552)     # scans to run -- NP in water in focus
scans = range(7587,7597)    # scans to run -- NP in water/DMSO in focus
#scans = range(7607,7617)   # scans to run -- NP in water/DMSO 1000um out
#scans = range(7637,7667)   # scans to run -- NP in water/DMSO 2000um out
exp_time = 5e-6             # exposure time [s] -- use for dmso-water samples
#exp_time = 5e-7            # exposure time [s] -- use for water sample
downsample = 10             # pixel binning
center_x, center_y = 1069.5, 201.5 # tristan detector center
print(sample, scans, exp_time, 'downsample:', downsample)

# define time delays
dlin = np.arange(1, 100) # 100 first data points lin-spaced
dlog = np.logspace(2,5,300).astype(int) # the rest log-spaced
delays = np.concatenate((dlin, dlog), axis=None)

width = 1   # width of rois/Q-bins
roi_min, roi_max = 10//downsample//width, 200//downsample//width # choose rois/Q-bins
print('rois:',roi_min, roi_max, width)
####################################################

# experimental parameters
distance = 4.0 # m
photon_energy = 10.0e3 #eV
h = 4.135667516e-15
c = 2.99792458e8
wavelength = h*c / photon_energy
pix = 55e-6 # m
center_x, center_y = 1069.5, 201.5 # tristan detector center

# make the single-pixel roi
roi_img = xpcs.make_mask()
roi_img = 1-roi_img
frames = xpcs.make_frames('/data/visitors/nanomax/20210623/2021052608/raw/sample/tristan-%04d*' %scans[0], 1, roi_img, 1)
img = xpcs.make_img(frames[0]) # ph/s 

# make simple radial Q-map
x = np.arange(0,roi_img.shape[1])
y = np.arange(0,roi_img.shape[0])
xx,yy = np.meshgrid(x,y)
r = (((xx-center_x)**2 + (yy-center_y)**2 ) ** 0.5)
rs = r*pix
tth = np.arctan2(rs, distance)
Q = 4.0e-9 * np.pi / wavelength * np.sin(0.5*tth) # 1/nm

####################################################

# additional masking 
roi_img_downsample = xpcs.downsample_img(roi_img, downsample)
roi_img_downsample[roi_img_downsample != roi_img_downsample.max()] = 0
roi_img_downsample[roi_img_downsample == roi_img_downsample.max()] = 1

# make downsampled roi based on the Q-map
Q_downsample = xpcs.downsample_img(Q, downsample)/downsample**2
roi_img_qmap, q = functions.make_rois_qmap(Q_downsample, mask=roi_img_downsample, nbins=roi_img.shape[1]//downsample//width, roi_min=roi_min, roi_max=roi_max, threshold=0, downsample=downsample)

# reshape roi
roi_img = np.zeros((515, 2069)), dtype=np.uint8)
roi_img[0:515//downsample, 0:2069//downsample] = roi_img_qmap

print('Q =', q)
print('roi created')

###################################################

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def worker(arg, roi_img, exp_time, downsample, delays):
    i, scan = arg
    frames = xpcs.make_frames('/data/visitors/nanomax/20210623/2021052608/raw/sample/tristan-%04d*' %scan, 
                         exp_time, roi_img, downsample)
    print(scan)
    return xpcs.calc_g2(frames, roi_img, delays)

pool = Pool(32)
g2 = pool.map(partial(worker, roi_img=roi_img, exp_time=exp_time, downsample=downsample, delays=delays), enumerate(scans))
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

print('Data saved')