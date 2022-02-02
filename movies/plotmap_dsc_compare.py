#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import healpy as hp
import h5py

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FixedLocator
from matplotlib.colors import PowerNorm

from argparse import ArgumentParser
from glob import glob
from progressbar import ProgressBar
import json

import hputils


from maptools import LOSDifferencer


#def load_discrete_los(fname):
#    res = []
#    
#    with h5py.File(fname, 'r') as f:
#        for key in f.keys():
#            nside,pix_idx = [int(s) for s in key.split()[1].split('-')]
#            
#            dset = f[key + '/discrete-los']
#            
#            #E_0, dm_0 = dset.attrs['min'][:]
#            #E_1, dm_1 = dset.attrs['max'][:]
#            #n_E, n_dm = dset.attrs['nPix'][:]
#            E_0, dm_0 = 0., 4.
#            E_1, dm_1 = 7., 19.
#            n_E, n_dm = 700, 120
#            dE = (E_1 - E_0) / n_E
#            
#            E = dset[0,2:,2:] * dE
#            
#            dm = np.linspace(dm_0, dm_1, n_dm+1)[:-1]
#            
#            res.append(((nside, pix_idx), E, dm))
#    
#    n_pix = len(res)
#    
#    nside = np.empty(n_pix, dtype='i4')
#    pix_idx = np.empty(n_pix, dtype='i8')
#    E = np.empty((n_pix,) + res[0][1].shape, dtype='f8')
#    
#    for k,row in enumerate(res):
#        nside[k], pix_idx[k] = row[0]
#        E[k] = row[1]
#    
#    dm = res[0][2]
#    
#    return nside, pix_idx, E, dm


def load_discrete_los(fname, thin_samples=None, diff=False):
    res = []
    
    E_0, dm_0 = 0., 4.
    E_1, dm_1 = 7., 19.
    n_E, n_dm = 700, 120
    dE = (E_1 - E_0) / n_E
    dm = np.linspace(dm_0, dm_1, n_dm+1)[:-1]
    print(fname)
    with h5py.File(fname, 'r') as f:
        if '/samples' in f:
            if thin_samples is None:
                E = f['/samples'][:,:,:]
            else:
                E = f['/samples'][:,:thin_samples,:]
            
            dset = f['/pixel_info'][:]
            nside = dset['nside'][:]
            pix_idx = dset['healpix_index'][:]
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


def load_all_pixels(fname_patterns):
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
        data = load_discrete_los(fn)
        nside.append(data[0])
        pix_idx.append(data[1])
        E.append(data[2])
        dm = data[3]
        bar.update(k+1)
    
    nside = np.concatenate(nside)
    pix_idx = np.concatenate(pix_idx)
    E = np.concatenate(E)
    
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


def interpolate_map(E, dm_range, dm):
    if dm >= dm_range[-1]:
        return E[...,-1]
    elif dm <= dm_range[0]:
        return 10**(0.2*(dm_range[0]-dm)) * E[...,0]
    
    # Linear interpolation coefficients
    k1 = np.searchsorted(dm_range, dm)
    k0 = k1 - 1
    
    #print('')
    #print(dm_range[-5:])
    #print(dm.size)
    #print(dm, k0)
    #print('{} < {} < {}'.format(dm_range[k0], dm, dm_range[k1]))
    
    d0 = dm - dm_range[k0]
    d1 = dm_range[k1] - dm
    a0 = d1 / (d0 + d1)
    a1 = 1. - a0
    
    #print('a0, a1 = {}, {}'.format(a0, a1))
    
    return a0*E[...,k0] + a1*E[...,k1]


def select_random_sample(E):
    n_pix, n_samples = E.shape[:2]
    idx0 = np.arange(n_pix)
    idx1 = np.random.randint(0, n_samples, size=n_pix)
    return E[idx0, idx1]


def get_sample_median(E):
    return np.median(E, axis=1)


def main():
    parser = ArgumentParser(
        description='Plot a map of discrete-los bayestar results.',
        add_help=True)
    parser.add_argument(
        '-i1', '--input1',
        type=str,
        nargs='+',
        required=True,
        help='First set of Bayestar output files.')
    parser.add_argument(
        '-i2', '--input2',
        type=str,
        nargs='+',
        required=True,
        help='Second set of Bayestar output files.')
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Plot filename. Format string taking frame number.')
    parser.add_argument(
        '-l1', '--label1',
        type=str,
        default='',
        help='Label for first set of outputs.')
    parser.add_argument(
        '-l2', '--label2',
        type=str,
        default='',
        help='Label for second set of outputs.')
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
        '--gamma',
        type=float,
        default=1.0,
        help='Power-law stretch to apply to reddening images.')
    parser.add_argument(
        '--vmax-diff',
        type=float,
        default=None,
        help='Maximum difference in reddening to plot.')
    parser.add_argument('--bounds', '-b', type=float, nargs=4, default=None,
                                         help='Bounds of pixels to plot (l_min, l_max, b_min, b_max).')
 
    args = parser.parse_args()
    
    if args.label1.startswith('$') and args.label2.startswith('$'):
        diff_label = args.label2[:-1] + ' \, - \, ' + args.label1[1:]
    else:
        diff_label = args.label2 + ' - ' + args.label1
    
    # Load data
    data = []
    for k,fn in enumerate([args.input1, args.input2]):
        print('Loading bayestar output files (set {} of 2)...'.format(k+1))
        nside, pix_idx, E, dm_edges = load_all_pixels(fn)
        data.append({
            'nside': nside,
            'pix_idx': pix_idx,
            'E': E,
            'dm_edges': dm_edges
        })
    
    # Generate rasterizer
    differencer = LOSDifferencer(
        data[0]['nside'],
        data[0]['pix_idx'],
        data[1]['nside'],
        data[1]['pix_idx']
    )
    
    print('Generating map rasterizer ...')
    proj_kw = json.loads(args.proj_kwargs)
    proj = get_projection(args.projection, **proj_kw)
    l0, b0 = args.lb_center
    img_shape = args.img_shape
    rasterizer = differencer.gen_rasterizer(
        img_shape,
        proj=proj,
        clip=True,
        l_cent=l0,
        b_cent=b0
    )
    
    # Create figure
    figsize = args.figsize
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = []
    im = []
    for i in range(3):
        #ax.append(fig.add_subplot(1,3,i+1))
        ax.append(fig.add_subplot(3,1,i+1))

        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        im.append(None)
    
    label_colors = ('b', 'orange')
    for i,(l,c) in enumerate(zip([args.label1, args.label2], label_colors)):
        ax[i].set_title(l, color=c, fontsize=13)
    
    ax[2].set_title(diff_label, color='white', fontsize=13)
    
    # Make room for other axes
    cbar_w = 0.025
    ts_h = 0.075
    
    bottom_margin = 0.1
    ts_margin = 0.05
    sp_bottom = bottom_margin + ts_h + ts_margin
    
    left_margin = 0.1
    cbar_margin = 0.00
    sp_left = left_margin + cbar_w + cbar_margin
    sp_top = 0.90
    
    fig.subplots_adjust(
        bottom=sp_bottom, top=sp_top,
        left=sp_left, right=1.-sp_left,
        wspace=0.03, hspace=0.03)
    
    # Time series
    ax_ts = fig.add_axes([
        left_margin, bottom_margin,
        1.-2.*left_margin, ts_h
    ])
    ax_ts2 = ax_ts.twinx()
    ax_ts_dist = ax_ts.twiny()
    
    med_ts = [None, None, None]
    hl_ts = [None, None, None]
    
    # Colorbars
    cax_l = fig.add_axes([
        left_margin, sp_bottom,
        cbar_w, sp_top-sp_bottom
    ])
    cax_r = fig.add_axes([
        1.-left_margin-cbar_w, sp_bottom,
        cbar_w, sp_top-sp_bottom
    ])
    
    # Rasterize at different distances
    print('Generating images ...')
    dm0, dm1 = args.dm_range
    dm_range = np.linspace(dm0, dm1, args.n_frames)
    
    vmin = 0.
    vmax = args.vmax
    if vmax is None:
        vmax = 1.2 * np.percentile(E[...,-1], 99.8)
        print('vmax = {}'.format(vmax))
    
    vmax_diff = args.vmax_diff
    if vmax_diff is None:
        vmax_diff = 0.25 * vmax
        print('vmax_diff = {}'.format(vmax_diff))
    
    # shape = (left/right/diff, low/med/high, frame)
    E_ts = np.full((3, 3, args.n_frames), np.nan)
    
    # Calculate distant limit of reddening
    E0 = []
    for d in data:
        pix_val_tmp = interpolate_map(d['E'], d['dm_edges'], 100.)
        E0.append(get_sample_median(pix_val_tmp))
        #print('')
        #for n,i,E in zip(d['nside'], d['pix_idx'], E0[-1]):
        #    if E < 0.2:
        #        print('E_inf({}-{}) = {:.3f} mag'.format(n,i,E))
        #print('')
        #pix_val_tmp = interpolate_map(d['E'], d['dm_edges'], 11.)
        #pix_val_tmp = get_sample_median(pix_val_tmp)
        #for n,i,dE in zip(d['nside'], d['pix_idx'], E0[-1]-pix_val_tmp):
        #    if dE > 0.2:
        #        print('dE({}-{}) = {:.3f} mag'.format(n,i,dE))
        #print('')
    E0.append(vmax_diff)
    
    bar = ProgressBar(max_value=len(dm_range))
    bar.update(0)
    
    for k,dm in enumerate(dm_range):
        img = []
        pix_val = []
        for i,d in enumerate(data):
            pix_val_tmp = interpolate_map(d['E'], d['dm_edges'], dm)
            pix_val.append(select_random_sample(pix_val_tmp))
            E = differencer.get_pix_val(i, pix_val[-1])
            img.append(rasterizer(E))
            
            E_ts[i,:,k] = np.nanpercentile(pix_val[i]/E0[i], [16., 50., 84.])
        
        dE = differencer.pix_diff(pix_val[0], pix_val[1])
        img.append(rasterizer(dE))
        
        E_ts[2,:,k] = np.nanpercentile(dE, [16., 50., 84.]) #/ E0[2]
        
        if k == 0:
            # Create elements of figure for first time
            for i,c in enumerate(label_colors):
                im[i] = ax[i].imshow(
                    img[i].T,
                    origin='lower',
                    interpolation='nearest',
                    norm=PowerNorm(args.gamma, vmin=vmin, vmax=vmax),
                    cmap='inferno',
                    extent = [args.bounds[1],args.bounds[0],args.bounds[2],args.bounds[3]])
                ax[i].set_aspect('equal')
                med_ts[i], = ax_ts.plot(
                    dm_range,
                    E_ts[i,1],
                    ls='-',
                    lw=1.,
                    alpha=0.7,
                    c=c)
            
            im[2] = ax[2].imshow(
                img[2].T,
                origin='lower',
                interpolation='nearest',
                vmin=-vmax_diff,
                vmax=vmax_diff,
                cmap='coolwarm_r',
                extent = [args.bounds[1],args.bounds[0],args.bounds[2],args.bounds[3]])
            ax[2].set_aspect('equal')
            
            med_ts[2], = ax_ts2.plot(
                dm_range,
                E_ts[2,1],
                ls='-',
                lw=1.,
                alpha=0.7,
                c='k')
            
            ax_ts.set_xlim(dm0, dm1)
            ax_ts.xaxis.set_major_locator(MultipleLocator(2.0))
            ax_ts.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax_ts.set_ylim(0., 1.2)
            
            #ax_ts2.set_xlim(dm0, dm1)
            ax_ts2.set_ylim(-1.2*vmax_diff, 1.2*vmax_diff)
            
            dm_ticks_maj = np.arange(5., 20.1, 5.)
            dm_labels = [r'${} \ \mathrm{{kpc}}$'.format(d) for d in ['0.1', '1', '10', '100']]
            dm_ticks_min = []
            for dm_upper in dm_ticks_maj:
                dm_ticks_min.append(dm_upper + 5. * np.log10(np.arange(0.2, 1.0, 0.1)))
            dm_ticks_min = np.hstack(dm_ticks_min)
            ax_ts_dist.set_xticks(dm_ticks_maj)
            ax_ts_dist.set_xticklabels(dm_labels)
            ax_ts_dist.xaxis.set_minor_locator(FixedLocator(dm_ticks_min))
            ax_ts_dist.set_xlim(dm0, dm1)
            
            ax_ts.set_xlabel(
                r'$\mu \ \left( \mathrm{mag} \right)$',
                fontsize=12)
            ax_ts.set_ylabel(
                r'$\mathrm{E} / \mathrm{E}_{d \rightarrow \infty}$',
                fontsize=12)
            ax_ts2.set_ylabel(
                r'$\Delta E \ \left( \mathrm{mag} \right)$',
                fontsize=12)
            
            cbar_l = fig.colorbar(im[0], cax=cax_l)
            cbar_l.set_label(r'$\mathrm{E} \left( B - V \right)$', fontsize=13)
            cax_l.yaxis.set_ticks_position('left')
            cax_l.yaxis.set_label_position('left')
            
            cbar_r = fig.colorbar(im[2], cax=cax_r)
            cbar_r.set_label(r'$\Delta \mathrm{E} \left( B - V \right)$', fontsize=13)
            
        else:
            # Update elements of figure
            for i in range(3):
                im[i].set_array(img[i].T)
            
            for i in range(3):
                med_ts[i].set_ydata(E_ts[i,1])
                hl_ts[i].remove()
        
        for i,c in enumerate(label_colors):
            hl_ts[i] = ax_ts.fill_between(
                dm_range,
                E_ts[i,0],
                E_ts[i,2],
                alpha=0.15,
                color=c)
        
        hl_ts[2] = ax_ts2.fill_between(
            dm_range,
            E_ts[2,0],
            E_ts[2,2],
            alpha=0.15,
            color='k')
        
        # Update title
        dist = 10.**(dm/5. + 1.)
        title = r'$\mu = {:.2f} \, \mathrm{{mag}}, \ d = {:.0f} \, \mathrm{{pc}}$'.format(dm, dist)
        fig.suptitle(title, fontsize=15)

        fig.subplots_adjust(hspace=0.025,wspace=0.025)
        
        # Save figure
        plot_fname = args.output.format(k)
        fig.savefig(plot_fname, dpi=150)
        
        bar.update(k+1)
    
    return 0


if __name__ == '__main__':
    main()

