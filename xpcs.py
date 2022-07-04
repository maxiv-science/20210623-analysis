import glob
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit
from numba.core import types
from numba.typed import List, Dict

tristan_shape = (515, 2069)
shutter_open_header = 0x0840
shutter_close_header = 0x0880
timing_resolution_fine = 1.5625e-9

@njit  
def make_hist(frames, values, frame_number, roi_img, downsample):
    for i, v in enumerate(values):
        x = (v >> 13) & 0x1FFF
        y = v & 0x1FFF
        
        # bin pixels
        x = x // downsample
        y = y // downsample
        v = (x << 13) + y
        
        if roi_img[y, x] == 0:
            continue
        frame = frames[frame_number[i]]
        if v in frame:
            frame[v] += 1 
        else:
            frame[v] = 1 
            
            
def make_mask():
    #256*256 3*3 pixels in the gap mask 5*5
    mask = np.zeros((515, 2069), dtype=np.int32)
    mask[255:260] = 1
    for i in range(1, 8):
        start = i*256 + (i-1)*3 - 1
        end = i*256 + i*3 + 1
        #print(start, end)
        mask[:, start:end] = 1

    # broken column on the right   
    mask[75:220, 1562] = 1

    # beamstop
    mask[:276, 1010:1120] = 1
    mask[:290, 1010:1120] = 1 # S: modified

    # beam streaks
    mask[150:250, :] = 1
    return mask

def downsample_img(img, n):
    strided = as_strided(img,
            shape=(img.shape[0]//n, img.shape[1]//n, n, n),
            strides=((img.strides[0]*n, img.strides[1]*n)+img.strides))
    return strided.sum(axis=-1).sum(axis=-1)

def make_roi(center_x, center_y, rois, downsample):
    mask = make_mask()
    mask = downsample_img(mask, downsample)
    nx = tristan_shape[1] // downsample
    ny = tristan_shape[0] // downsample
    x = np.arange(nx)
    y = np.arange(ny) 
    X, Y = np.meshgrid(x, y)
    pixel_index = (Y * nx) + X
    roi_img = np.zeros(tristan_shape, dtype=np.uint8)
    R2 = (X - center_x // downsample )**2 + (Y - center_y // downsample)**2
    for i, roi in enumerate(rois):
        ind = np.where((R2 >= roi[0]**2) & (R2 <= roi[1]**2) & (mask==0))
        ind = np.unravel_index(pixel_index[ind], (ny, nx))
        roi_img[ind] = i+1
    return roi_img
    
def make_frames(file_pattern, frame_duration, roi_img, downsample):
    fnames = glob.glob(file_pattern)

    shutter_open_counts = -1
    shutter_close_counts = -1
    for fname in fnames:
        with h5py.File(fname, 'r') as fh:
            cue_id = fh['cue_id'][:]
            cue_timestamp_zero = fh['cue_timestamp_zero'][:]
        index, = np.where(cue_id == shutter_open_header)
        if len(index) > 0:
            shutter_open_counts = cue_timestamp_zero[index[0]]

        index, = np.where(cue_id == shutter_close_header)
        if len(index) > 0:
            shutter_close_counts = cue_timestamp_zero[index[0]]

    run_time = (shutter_close_counts - shutter_open_counts) * timing_resolution_fine
    number_of_frames = int(run_time / frame_duration) + 1
    #print('run_time', run_time)
    #print('number_of_frames', number_of_frames)
    frame_counts = int(frame_duration / timing_resolution_fine)
    frames = List()
    for i in range(number_of_frames):
        frames.append(Dict.empty(
        key_type=types.uint32,
        value_type=types.uint32,
    ))
    
    for fname in fnames:
        with h5py.File(fname, 'r') as fh:
            event_time_offset = fh['event_time_offset'][:]
            event_id = fh['event_id'][:]
        event_time_absolute = event_time_offset - shutter_open_counts
        frame_number = (event_time_absolute / frame_counts).astype('u4')
        argfilter = (event_time_offset > shutter_open_counts) & (event_time_offset < shutter_close_counts)
        event_id = event_id[argfilter]
        frame_number = frame_number[argfilter]
        #print(fname)
        make_hist(frames, event_id, frame_number, roi_img, downsample)
    return frames

@njit
def make_img(frame):
    img = np.zeros(tristan_shape, dtype=np.uint16)
    for k, v in frame.items():
        x = (k >> 13) & 0x1FFF
        y = k & 0x1FFF 
        img[y, x] = v
    return img

@njit
def g2_denominator(frames, roi_img, npixels):
    nq = len(npixels)
    values = np.zeros(nq)
    for t in range(len(frames)):
        for k, v in frames[t].items():
            x = (k >> 13) & 0x1FFF
            y = k & 0x1FFF 
            qbin = roi_img[y, x] - 1
            values[qbin] += v

    values /= len(frames) * npixels
    return values**2

@njit
def g2_nominator(frames, roi_img, tau, npixels): #, square=False):
    correlation = np.zeros(len(npixels))
    nt = len(frames) - tau           # no of initial times (frames) t0
    for t in range(len(frames)-tau): # summing over initial times (frames) t0
        f2 = frames[t+tau]
        for k, v1 in frames[t].items(): # summing over pixels 
            if k in f2:
                x = (k >> 13) & 0x1FFF
                y = k & 0x1FFF 
                qbin = roi_img[y, x] - 1
                correlation[qbin] += v1 * f2[k]
    return correlation / nt / npixels

@njit
def ttc_inner_prod(frames, roi_img, npixels):
    """
    The product (I_{t1} * I_{t2}) averaged over all pixels in each q-bin.
    
    Returns an array (qbin, t1, t2)
    """
    qbins = np.unique(roi_img)[1:]
    shape = (len(qbins), len(frames), len(frames))
    result = np.zeros(shape, dtype=np.float32)
    for t1 in range(len(frames)):
        for t2 in range(t1+1):
            for k1, v1 in frames[t1].items():
                if k1 in frames[t2]:
                    x = (k1 >> 13) & 0x1FFF
                    y = k1 & 0x1FFF 
                    q = roi_img[y, x] - 1
                    result[q, t1, t2] += v1 * frames[t2][k1] # one triangle of the ttc
                    if not t1 == t2:
                        result[q, t2, t1] += v1 * frames[t2][k1] # other triangle of the tcc
    result[:] = result / npixels.reshape((len(qbins), 1, 1))
    return result

@njit
def ttc_Iav(frames, roi_img, npixels, square=False):
    """
    The array I_{t} or I_{t}**2 averaged over all pixels in each q-bin
    
    Returns an array (qbin, t)
    """
    qbins = np.unique(roi_img)[1:]
    shape = (len(qbins), len(frames))
    result = np.zeros(shape, dtype=np.float32)
    for t in range(len(frames)):
        for k, v in frames[t].items():
            x = (k >> 13) & 0x1FFF
            y = k & 0x1FFF 
            q = roi_img[y, x] - 1
            if square:
                result[q, t] += v**2
            else:
                result[q, t] += v
    result[:] = result / npixels.reshape((len(qbins), 1))
    return result

def calc_ttc(frames, roi_img):
    """
    according to Perakis and Gutt, PCCP, 2020: 
    
    <I_{t1} * I_{t2}>
  ----------------------
    <I_{t1}><I_{t2}>

    where <> is the average over all pixels in a q-bin.
    """    
    _, npixels = np.unique(roi_img, return_counts=True)
    npixels = npixels[1:]

    # do the heavy work:
    crossterm = ttc_inner_prod(frames, roi_img, npixels)
    Iav = ttc_Iav(frames, roi_img, npixels)
    I2av = ttc_Iav(frames, roi_img, npixels, square=True)

    # reshape to represent (t1, t2) - they're symmetric
    nq = Iav.shape[0]
    nt = Iav.shape[1]
    nominator = crossterm 
    denominator = Iav.reshape((nq,nt,1))*Iav.reshape((nq,1,nt))
    return nominator / denominator

def calc_g2(frames, roi_img, delays):
    _, npixels = np.unique(roi_img, return_counts=True)
    npixels = npixels[1:]
    den = g2_denominator(frames, roi_img, npixels)
    res = []
    for tau in delays:
        res.append(g2_nominator(frames, roi_img, tau, npixels) / den)
    return np.array(res)


if __name__ == '__main__':
    # example usage
    path = './'
    scan = 7563 # hematite
    downsample, dt = 5, 1e-3
    q_offset, q_width, q_number = 100//downsample, 10//downsample, 5

    # loading
    q_bins = [(q_offset+q_width*i, q_offset+q_width*(i+1)) for i in range(q_number)]
    mask = make_mask()
    center_x, center_y = 1069.5, 201.5
    roi_img = make_roi(center_x, center_y, q_bins, downsample)
    frames = make_frames(path+'tristan-%04u*'%scan, dt, roi_img, downsample)

    # g2 calculation
    delays = (np.logspace(np.log10(dt), 1, 50) / dt).astype(int)
    g2 = calc_g2(frames, roi_img, delays)
    import matplotlib.pyplot as plt; plt.ion()
    plt.plot(np.log10(delays*dt), g2-1)

    # ttc calculation and plot
    ttc = calc_ttc(frames[:1000], roi_img)
    plt.figure()
    plt.imshow(ttc[0])
    plt.title('scan %u, downsampling x %u'%(scan,downsample))
    plt.xlabel('time with dt = %.1e'%dt)
    plt.colorbar()