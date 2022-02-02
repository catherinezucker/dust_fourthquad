#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import h5py
import transforms3d

from glob import glob
from progressbar import ProgressBar

from projection_tools import OrthographicProjection, MapData, MapDataHires


def load_discrete_los(fname, thin_samples=None, diff=False, reduction=None):
    res = []
    
    E_0, dm_0 = 0., 4.
    E_1, dm_1 = 7., 19.
    n_E, n_dm = 700, 120
    dE = (E_1 - E_0) / n_E
    dm = np.linspace(dm_0, dm_1, n_dm+1)[:-1]
    
    with h5py.File(fname, 'r') as f:
        if '/discrete-los' in f:
            if thin_samples is None:
                E = f['/discrete-los'][:,2:,2:]
            else:
                E = f['/discrete-los'][:,2:thin_samples+2,2:]
            
            if reduction is not None:
                E = reduction(E)
                E.shape = (E.shape[0], 1, E.shape[1])
            
            dset = f['/pix_info'][:]
            nside = dset['nside'][:]
            pix_idx = dset['pix_idx'][:]
        else:
            for key in f.keys():
                nside,pix_idx = [int(s) for s in key.split()[1].split('-')]
                
                dset = f[key + '/discrete-los']
                
                #E_0, dm_0 = dset.attrs['min'][:]
                #E_1, dm_1 = dset.attrs['max'][:]
                #n_E, n_dm = dset.attrs['nPix'][:]
                E = dset[0,2:,2:] * dE
                
                res.append(((nside, pix_idx), E, dm))
                
                n_pix = len(res)
                
                nside = np.empty(n_pix, dtype='i4')
                pix_idx = np.empty(n_pix, dtype='i8')
                E = np.empty((n_pix,) + res[0][1].shape, dtype='f8')
                
                for k,row in enumerate(res):
                    nside[k], pix_idx[k] = row[0]
                    E[k] = row[1]
                
                dm = res[0][2]
    
    if diff:
        for k in range(n_dm-1, 0, -1):
            E[...,k] -= E[...,k-1]
        
        d = 10.**(dm/5. - 2.)
        E[...,0] /= d[0]
        for k in range(1,n_dm):
            E[...,k] /= d[k] - d[k-1]
    
    return nside, pix_idx, E, dm


def load_all_pixels(fname_patterns, **kwargs):
    fnames = []
    for fn_pattern in fname_patterns:
        fnames += glob(fn_pattern)
    
    nside = []
    pix_idx = []
    E = []
    dm = None
    
    bar = ProgressBar(max_value=len(fnames))
    bar.update(0)
    for k,fn in enumerate(fnames):
        data = load_discrete_los(fn, **kwargs)
        nside.append(data[0])
        pix_idx.append(data[1])
        E.append(data[2])
        dm = data[3]
        bar.update(k+1)
    
    nside = np.concatenate(nside)
    pix_idx = np.concatenate(pix_idx)
    E = np.concatenate(E)
    
    return nside, pix_idx, E, dm


def multires2hires(nside, pix_idx, pix_val, fill=np.nan):
    nside_unique = np.unique(nside)
    nside_max = np.max(nside_unique)
    
    pix_val_hires = np.full(hp.pixelfunc.nside2npix(nside_max), fill)
    
    for n in nside_unique:
        # Get indices of all pixels at current nside level
        idx = (nside == n)

        # Determine nested index of each selected pixel in upsampled map
        mult_factor = (nside_max//n)**2
        pix_idx_n = pix_idx[idx] * mult_factor

        # Write the selected pixels into the upsampled map
        pix_val_n = pix_val[idx]
        for offset in range(mult_factor):
            pix_val_hires[pix_idx_n+offset] = pix_val_n[:]
    
    return pix_val_hires


def downsample_nside(pix_val, n_levels=1):
    n = (2**n_levels)**2
    v = np.reshape(pix_val, (pix_val.size//n, n))
    return np.nanmean(v, axis=1)


def convert_to_hires(it, diff=True, reduction='mean'):
    terra_dir = '/n/fink2/ggreen/bayestar/terra'
    fn_base = '{0}/output/combined/it{1}/it{1}'.format(terra_dir, it)
    
    if reduction == 'mean':
        f = lambda v: np.mean(v, axis=1)
    elif reduction == 'median':
        f = lambda v: np.mean(v, axis=1)
    elif reduction == 'sample':
        def f(v):
            n_s = v.shape[1]
            return v[:, np.random.randint(n_s), :]
    else:
        raise ValueError("Unknown reduction: '{}'".format(reduction))
    
    # Load data
    print('Loading bayestar output files ...')
    fname = [fn_base + '_combined.*.h5']
    nside, pix_idx, pix_val, dm_edges = load_all_pixels(
        fname,
        diff=diff,
        reduction=f
    )
    
    print('Creating MapData object ...')
    map_data = MapData(nside, pix_idx, pix_val, dm_edges[0], dm_edges[-1])
    
    print('Getting hi-res map ...')
    pix_val_hires = map_data.get_hires_map()[:,0,:].astype('f4')
    
    # Save hi-res map
    print('Saving ...')
    suffix = ['_cumulative', '_diff'][int(diff)]
    np.save(fn_base + suffix + '_{}.npy'.format(reduction),
            pix_val_hires)
    
    print('Saving low-res map ...')
    s = (pix_val_hires.shape[0]//64, 64, pix_val_hires.shape[1])
    
    if reduction == 'mean':
        pix_val_lowres = np.mean(np.reshape(pix_val_hires, s), axis=1)
    elif reduction == 'median':
        pix_val_lowres = np.median(np.reshape(pix_val_hires, s), axis=1)
    elif reduction == 'sample':
        pix_val_lowres = np.reshape(pix_val_hires, s)[:,0,:]
    
    np.save(fn_base + suffix + '_lowres_{}.npy'.format(reduction),
            pix_val_lowres)


def main():
    for reduction in ['median']:
        for it in [4]:
            convert_to_hires(it, reduction=reduction, diff=False)
            convert_to_hires(it, reduction=reduction, diff=True)
    
    return 0


if __name__ == '__main__':
    main()

