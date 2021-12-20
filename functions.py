import numpy as np
import sys
import os
import xpcs

 # parameters
distance = 4.0 # m
pix = 55e-6 # m
photon_energy = 10.0e3 #eV
h = 4.135667516e-15
c = 2.99792458e8
wavelength = h*c/photon_energy
center_x, center_y = 1069.5, 201.5 # tristan center
center = center_y, center_x
    

def calc_Iq(Q_map, I_map, mask, downsample, nbins, n_max, n_min, threshold):
   
    # --- calculate I(q) based on a Q-map
    
    # roi
    smallx1 = np.int((center_x-n_max)//downsample)
    smallx2 = np.int((center_x+n_max)//downsample)
    smally1 = np.int((center_y-n_min)//downsample)
    smally2 = np.int((center_y+n_max)//downsample)
    
    # mask
    if mask is None:
        mask=np.ones(Q_map.shape) # unitary mask
    Q_map = Q_map*mask
    Q_map = Q_map[smally1:smally2,smallx1:smallx2]
    I_map = I_map*mask
    I_map = I_map[smally1:smally2,smallx1:smallx2]
    
    # flatten 2D maps
    Q_map_flat = Q_map.flatten()
    I_map_flat = I_map.flatten()

    # sort into bins with similar intensity
    ind = np.argsort(Q_map_flat)
    Q_map_flat = Q_map_flat[ind]
    I_map_flat = I_map_flat[ind]

    # Q-bins
    edges = np.linspace(Q_map_flat[Q_map_flat>threshold].min(), Q_map_flat.max(), nbins+1)
    #print(edges[0], edges[-1])

    # group indices in different Q-bins
    inds = np.digitize(Q_map_flat, edges)
    Q_av = np.array([Q_map_flat[inds == i].mean() for i in range(0, nbins+1)])
    I_av = np.array([I_map_flat[inds == i].mean() for i in range(0, nbins+1)])

    return Q_av, I_av



def calc_q1d(Q_map, mask, downsample, nbins, n_max, n_min, threshold):
    
    # --- calculate the 1D Q based on a 2D Q-map
    
    # roi
    smallx1 = np.int((center_x-n_max)/downsample)
    smallx2 = np.int((center_x+n_max)/downsample)
    smally1 = np.int((center_y-n_min)/downsample)
    smally2 = np.int((center_y+n_max)/downsample)

    # mask
    if mask is None:
        mask=np.ones(Q_map.shape) # unitary mask
    Q_map = Q_map*mask
    Q_map = Q_map[smally1:smally2,smallx1:smallx2]

    # flatten 2D maps
    Q_map_flat = Q_map.flatten()

    # sort into bins with similar intensity
    ind = np.argsort(Q_map_flat)
    Q_map_flat = Q_map_flat[ind]

    # Q-bins
    edges = np.linspace(Q_map_flat[Q_map_flat>threshold].min(), Q_map_flat.max(), nbins+1)
    print(edges[0], edges[-1])

     # group indices in different Q-bins
    inds = np.digitize(Q_map_flat, edges)
    Q_av = np.array([Q_map_flat[inds == i].mean() for i in range(0, nbins+1)])
   
    return Q_av 




def make_rois_qmap(Q_map, mask, nbins, roi_min, roi_max, threshold, downsample): 

    # mask
    if mask is None:
        mask=np.ones(Q_map.shape) # unitary mask
    Q_map = Q_map*mask
    Q_map = Q_map 
   
    # flatten 2D maps
    Q_map_flat = Q_map.flatten()

    # sort into bins with similar intensity
    ind = np.argsort(Q_map_flat)
    Q_map_flat = Q_map_flat[ind]

    # Q-bins
    edges = np.linspace(Q_map_flat[Q_map_flat>threshold].min(), Q_map_flat.max(), nbins+1)
    print(edges[0], edges[-1], edges.shape)
    
    # group indices in different Q-bins
    rois = np.digitize(Q_map, edges)
    rois[(rois<roi_min) | (rois>roi_max)] = 0
    
    inds = np.digitize(Q_map_flat, edges)
    Q_av = np.array([Q_map_flat[inds == i].mean() for i in range(roi_min, roi_max+1)])
    print(len(Q_av))
    
    return rois, Q_av 



def shift_image(X, dx, dy):     # shift image and replace with zeros
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

