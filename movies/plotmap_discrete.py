#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import healpy as hp
import h5py

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FixedLocator
import matplotlib.cm as cm

from argparse import ArgumentParser
from glob import glob
from progressbar import ProgressBar
import json

import hputils

from astropy_healpix import healpy



def load_discrete_los(fname, thin_samples=None, diff=False):
    res = []
    
    E_0, dm_0 = 0., 4.
    E_1, dm_1 = 7., 19.
    n_E, n_dm = 700, 120
    dE = (E_1 - E_0) / n_E
    dm = np.linspace(dm_0, dm_1, n_dm+1)[:-1]
    
    with h5py.File(fname, 'r') as f:

        if '/samples' in f:
            if thin_samples is None:
                E = f['/samples'][:,:,:]
            else:
                E = f['/samples'][:,:thin_samples,:]
            
            dset = f['/pixel_info'][:]
            nside = dset['nside'][:]
            pix_idx = dset['pix_idx'][:]
            print(nside)
        elif '/discrete-los' in f:
            if thin_samples is None:
                E = f['/discrete-los'][:,2:,2:]
            else:
                E = f['/discrete-los'][:,2:thin_samples+2,2:]
            
            dset = f['/pix_info'][:]
            nside = dset['nside'][:]
            pix_idx = dset['pix_idx'][:]
        else:
            for key in f.keys():
                #if f[key].attrs['n_stars_rejected'] == f[key].attrs['n_stars']:
                #if f[key].attrs['reject_frac'] > 0.999:
                    # All stars were rejected => no LOS reddening was computed
                 #   continue
                
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

    fnames.sort()
    #for fname in fnames:
    #    fo = h5py.File(fname,'r')
    #    if 'discrete-los' not in fo:
    #        fnames.remove(fname)
    #    fo.close()
        
    #fnames.remove('/n/fink2/czucker/Plane_Final/output/it0/G314/G314.00312.h5')
    #fnames.remove('/n/fink2/czucker/Plane_Final/output/it0/G314/G314.00486.h5')
    #fnames.remove('/n/fink2/czucker/Plane_Final/output/it0/G314/G314.00748.h5')
    #fnames.remove('/n/fink2/czucker/Plane_Final/output/it0/G314/G314.00716.h5')
    #fnames.remove('/n/fink2/czucker/Plane_Final/output/it0/G314/G314.00790.h5')
    
    nside = []
    pix_idx = []
    E = []
    dm = None
    
    bar = ProgressBar(max_value=len(fnames))
    bar.update(0)
    for k,fn in enumerate(fnames):
        print(fn)
        data = load_discrete_los(fn, **kwargs)
        nside.append(data[0])
        pix_idx.append(data[1])
        E.append(data[2])
        dm = data[3]
        bar.update(k+1)
    
    nside = np.concatenate(nside)
    pix_idx = np.concatenate(pix_idx)
    E = np.concatenate(E)

    np.save('nside.npy',nside)
    np.save('pix_idx.npy',pix_idx)
    np.save('E.npy',E)
    np.save('dm.npy',dm)

    npix = healpy.nside2npix(1024)
    profiles = np.zeros((npix,120))
    profiles[pix_idx,:] = np.median(E,axis=1)

    np.save("it0_median_compiled.npy",profiles)
    
    return nside, pix_idx, E, dm


def get_projection(name, **kwargs):
    name = name.lower()
    if name == 'cartesian':
        return hputils.Cartesian_projection(**kwargs)
    elif name == 'hammer':
        return hputils.Hammer_projection(**kwargs)
    elif name == 'stereographic':
        return hputils.Stereographic_projection(**kwargs)
    elif name == 'gnomonic':
        return hputils.Gnomonic_projection(**kwargs)
    else:
        raise ValueError('Unknown map projection: "{:s}"'.format(name))


def interpolate_map(E, dm_range, dm, smooth=True):
    if smooth:
        if dm >= dm_range[-1]:
            return E[...,-1]
        elif dm <= dm_range[0]:
            return 10**(0.2*(dm_range[0]-dm)) * E[...,0]
        
        # Linear interpolation coefficients
        #k0 = np.searchsorted(dm_range, dm)
        #print(dm_range)
        #print(dm)
        #print(k0)
        #print('')
        #k1 = k0 + 1
        #d0 = dm - dm_range[k0]
        #d1 = dm_range[k1] - dm
        #a0 = d1 / (d0 + d1)
        #a1 = 1. - a0
        
        k1 = np.searchsorted(dm_range, dm)
        k0 = k1 - 1
        d0 = dm - dm_range[k0]
        d1 = dm_range[k1] - dm
        a0 = d1 / (d0 + d1)
        a1 = 1. - a0
        
        return a0*E[...,k0] + a1*E[...,k1]
    else:
        if dm >= dm_range[-1]:
            return E[...,-1]
        
        k1 = np.searchsorted(dm_range, dm)
        return E[...,k1-1]


def select_random_sample(E):
    n_pix, n_samples = E.shape[:2]
    idx0 = np.arange(n_pix)
    idx1 = np.random.randint(0, n_samples, size=n_pix)
    return E[idx0, idx1]


def average_over_samples(E):
    return np.nanmean(E, axis=1)


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


def main():
    parser = ArgumentParser(
        description='Plot a map of discrete-los bayestar results.',
        add_help=True)
    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='Bayestar output files.')
    parser.add_argument(
        'output',
        type=str,
        help='Plot filename. Format string taking frame number.')
    parser.add_argument(
        '--img-shape',
        type=int,
        nargs=2,
        default=(500,500),
        help='Shape of the image.')
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=(6., 6.),
        help='Figure width & height, in inches.')
    parser.add_argument(
        '--projection',
        type=str,
        default='Cartesian',
        help='Name of map projection to use.')
    parser.add_argument(
        '--proj-kwargs',
        type=str,
        default='{}',
        help='Projection keyword arguments, in JSON format.')
    parser.add_argument(
        '--lb-center',
        type=float,
        nargs=2,
        default=(0.,0.),
        help='Central (l,b) of projection.')
    parser.add_argument(
        '--dm-range',
        type=float,
        nargs=2,
        default=(4.,19.),
        help='Minimum, maximum distance moduli.')
    parser.add_argument('--l-lines', '-ls', type=float, nargs='+', default=None,
                                         help='Galactic longitudes at which to draw lines.')
    parser.add_argument('--b-lines', '-bs', type=float, nargs='+', default=None,
                                         help='Galactic latitudes at which to draw lines.')
    parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None,
                                         help='Bounds of pixels to plot (l_min, l_max, b_min, b_max).')
    parser.add_argument(
        '--linear-distance',
        action='store_true',
        help='Space frames linearly in distance (rather than distance modulus).')
    parser.add_argument(
        '--n-frames',
        type=int,
        default=121,
        help='# of images to generate (spaced evenly in DM).')
    parser.add_argument(
        '--vmax',
        type=float,
        default=None,
        help='Maximum reddening to plot.')
    parser.add_argument(
        '--density',
        action='store_true',
        help='Plot dE/ds, rather than cumulative E.')
    parser.add_argument(
        '--thin-samples',
        type=int,
        default=None,
        help='Limit the number of samples loaded.')
    parser.add_argument(
        '--asinh',
        type=float,
        default=None,
        help='Use an asinh stretch, with <x> as the turnover point.')
    parser.add_argument(
        '--reduction',
        type=str,
        default='sample',
        choices=('sample','mean'),
        help='How to deal with samples.')
    parser.add_argument(
        '--dpi',
        type=float,
        default=150,
        help='DPI of figure.')
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Figure title (can be a LaTeX string).')
    args = parser.parse_args()
    
    # Load data
    print('Loading bayestar output files ...')
    nside, pix_idx, E, dm_edges = load_all_pixels(
        args.input,
        diff=args.density,
        thin_samples=args.thin_samples
    )
    
    # Generate rasterizer
    print('Generating map rasterizer ...')
    proj_kw = json.loads(args.proj_kwargs)
    proj = get_projection(args.projection, **proj_kw)
    l0, b0 = args.lb_center
    img_shape = args.img_shape
    #nside_img = 128
    #npix_img = hp.pixelfunc.nside2npix(nside_img)
    rasterizer = hputils.MapRasterizer(
        nside,
        pix_idx,
        #nside_img * np.ones(npix_img, dtype='i4'),
        #np.arange(npix_img, dtype='i4'),
        img_shape,
        proj=proj,
        nest=True, clip=True,
        l_cent=l0, b_cent=b0
    )
    
    # Create figure
    figsize = args.figsize
    fig = plt.figure(figsize=figsize, dpi=args.dpi)
    gs = GridSpec(
        2, 2,
        width_ratios=(1,0.08),
        height_ratios=(1,0.08),
        wspace=0.05,
        hspace=0.2
    )
    #ax = fig.add_subplot(1,1,1)
    ax = fig.add_subplot(gs[0,0])
    cax = fig.add_subplot(gs[0,1])
    dax = fig.add_subplot(gs[1,0])
    #ax.set_xticks([]) 
    #ax.set_yticks([]) 
    if args.title is not None:
        ax.set_title(args.title)
    #ax.axis('off')
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.97)
    im = None

    dax.set_xlabel(r'$\mathrm{distance\ modulus}\ \left( \mathrm{mag} \right)$')
    dax.xaxis.set_major_locator(AutoLocator())
    dax.xaxis.set_minor_locator(AutoMinorLocator())
    dax.set_yticks([])
    
    dax2 = dax.twiny()
    #dax2.set_xlabel(r'$\mathrm{distance}\ \left( \mathrm{kpc} \right)$')
    dax2.set_yticks([])
    
    # Rasterize at different distances
    print('Generating images ...')
    dm0, dm1 = args.dm_range
    d_range = np.linspace(
        10.**(dm0/5.-2.),
        10.**(dm1/5.-2.),
        args.n_frames
    )
    if args.linear_distance:
        dm_range = 5. * (np.log10(d_range) + 2.)
        print(dm_range)
    else:
        dm_range = np.linspace(dm0, dm1, args.n_frames)
    
    vmin = 0.
    vmax = args.vmax
    if vmax is None:
        vmax = np.percentile(E, 99.5)
    if args.asinh is not None:
        vmax = np.arcsinh(vmax)
    print('vmax = {}'.format(vmax))
    
    bar = ProgressBar(max_value=len(dm_range))
    bar.update(0)
    
    cmap = cm.inferno
    #cmap.set_bad('lightgray', 1.0)
    
    for k,dm in enumerate(dm_range):
        E_dm = interpolate_map(E, dm_edges, dm, smooth=(not args.density))
        if args.reduction == 'sample':
            E_dm = select_random_sample(E_dm)
        elif args.reduction == 'mean':
            E_dm = average_over_samples(E_dm)
        else:
            raise ValueError('Invalid reduction: "{}"'.format(args.reduction))
        #E_dm = multires2hires(nside, pix_idx, E_dm)
        #E_dm = downsample_nside(E_dm, n_levels=3)
        
        if args.asinh is not None:
            E_dm = np.arcsinh(E_dm/args.asinh)
        
        img = rasterizer(E_dm)
        #img = np.ma.array(img, mask=np.isnan(img))
        #img = np.clip(img, vmin, vmax)
        
        if k == 0:
            im = ax.imshow(
                img.T,
                origin='lower',
                interpolation='nearest',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                extent = [args.bounds[1],args.bounds[0],args.bounds[2],args.bounds[3]])
            cb = fig.colorbar(im, cax=cax, orientation='vertical')
            if args.density:
                cb.set_label(r'$\mathrm{d}E / \mathrm{d}s$')
            else:
                cb.set_label(r'$E$')
            dfill = dax.axvspan(dm_range[0], dm)
            dax.set_xlim(dm_range[0], dm_range[-1])
            dax2.set_xlim(dm_range[0], dm_range[-1])
            dticks_min = np.hstack([
                np.arange(0.1, 1., 0.1),
                np.arange(1., 10., 1.0),
                np.arange(10., 100., 10.)
            ])
            dm_ticks_min = 5.*(np.log10(dticks_min)+2.)
            dax2.xaxis.set_minor_locator(FixedLocator(dm_ticks_min))
            dax2.xaxis.set_major_locator(FixedLocator([5., 10., 15.]))
            dax2.set_xticklabels([
                r'$100\,\mathrm{pc}$',
                r'$1\,\mathrm{kpc}$',
                r'$10\,\mathrm{kpc}$'
            ])
        else:
            im.set_array(img.T)
            dfill.remove()
            dfill = dax.axvspan(dm_range[0], dm)
        
        #dist = 10.**(dm/5. + 1.)
        #title = r'$\mu = {:.2f} \, \mathrm{{mag}}, \ d = {:.0f} \, \mathrm{{pc}}$'.format(dm, dist)
        #ax.set_title(title)

        ax.tick_params(axis='both',direction='out',pad=5)

        ax.set_xlim([args.bounds[1],args.bounds[0]])
        ax.set_ylim([args.bounds[2],args.bounds[3]])
        ax.set_xticks(args.l_lines)
        ax.set_yticks(args.b_lines)

        plot_fname = args.output.format(k)
        fig.savefig(plot_fname, dpi=args.dpi,bbox_inches='tight')
        
        bar.update(k+1)
    
    return 0


if __name__ == '__main__':
    main()

