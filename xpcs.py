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
def denominator(frames, roi_img, npixels):
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
def nominator(frames, roi_img, tau, npixels):
    correlation = np.zeros(len(npixels))
    nt = len(frames) - tau
    for t in range(len(frames)-tau):
        f2 = frames[t+tau]
        for k, v1 in frames[t].items():
            if k in f2:
                x = (k >> 13) & 0x1FFF
                y = k & 0x1FFF 
                qbin = roi_img[y, x] - 1
                correlation[qbin] += v1 * f2[k]
    return correlation / nt / npixels

def calc_g2(frames, roi_img, delays):
    _, npixels = np.unique(roi_img, return_counts=True)
    npixels = npixels[1:]
    den = denominator(frames, roi_img, npixels)
    res = []
    for tau in delays:
        res.append(nominator(frames, roi_img, tau, npixels) / den)
    return np.array(res)


