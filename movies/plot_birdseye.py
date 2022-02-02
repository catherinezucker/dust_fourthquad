#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import healpy as hp
import h5py
import transforms3d

from glob import glob
from progressbar import ProgressBar

from projection_tools import OrthographicProjection, MapData, MapDataHires
from bayestar_output_utils import convert_to_hires


def project_slices(it, scale, z_max):
    shape = (1000,1000)
    proj = OrthographicProjection(shape, scale).rotated(np.pi, 0., 0., 'syzx')

    print('Loading mean bayestar output ...')
    fname = 'it0_median_compiled.npy'.format(it)
    pix_val = np.load(fname)
    
    print('Creating MapData object ...')
    map_data = MapDataHires(pix_val, 4., 19.)
    
    # Project slices
    surf_density = np.zeros(shape, dtype='f8')
    n_finite = np.zeros(shape, dtype='i4')
    x = np.array([0., 0., z_max])
    
    for d in np.linspace(0., 2.*z_max, 4*int(z_max)+1):
        print('d = {}'.format(d))
        v = proj.get_surface(x, d)
        
        print('z = {} +- {} pc'.format(np.mean(v[2]), np.std(v[2])))
        
        print('Getting pixel values ...')
        pix_val_v = map_data.get_pix_val(v, outer_fill=np.nan)
        
        print('<rho> = {}'.format(np.nanmean(pix_val_v)))
        #pix_val_v = np.mean(pix_val_v, axis=2)
        
        idx = np.isfinite(pix_val_v)
        surf_density[idx] += pix_val_v[idx]
        n_finite += idx.astype('i4')
    
    surf_density /= n_finite.astype('f8')
    
    print('Saving arrays ...')
    fn_base = './topdown/birdseye_'.format(it)
    suffix = '_scale{:.0f}_z{:.0f}.npy'.format(scale, z_max)
    np.save(fn_base + 'surf_density' + suffix, surf_density)
    np.save(fn_base + 'n_finite' + suffix, n_finite)


def plot_birdseye(it, scale, z_max, finite_frac=0., vmax=1.5, it_label=[]):
    if not hasattr(it, '__len__'):
        it = [it]
    
    n_it = len(it)
    

    #w_max = 8.5 - 2.0
    #h_max = 11.0 - 2.0
    
    w_max = 13
    h_max = 18
    
    n_rows, n_cols = [
        (1,1),
        (1,2),
        (1,3),
        (2,2),
        (2,3)
    ][n_it-1]
    
    w = w_max
    h = w * (n_rows+0.25)/n_cols
    
    if h > h_max:
        a = h_max / h
        w *= a
        h *= a
    
    fig = plt.figure(figsize=(w,h), dpi=150)
    
    print(n_rows, n_cols)
    print(w,h)
    
    extent = 0.001*0.5*scale * np.array([-1., 1., -1., 1.])
    suffix = '_scale{:.0f}_z{:.0f}'.format(scale, z_max)
    
    left_label, right_label, top_label, bottom_label = False, False, False, False
    
    im = None
    
    for k,i in enumerate(it):
        fn_base = './topdown/birdseye_'.format(i)
        surf_density = np.load(fn_base + 'surf_density' + suffix + '.npy')
        n_finite = np.load(fn_base + 'n_finite' + suffix + '.npy')
        
        # Filter out regions with too few finite reddenings
        idx = (n_finite < finite_frac * np.max(n_finite))
        surf_density[idx] = np.nan
        
        # Plot surface density
        ax = fig.add_subplot(n_rows, n_cols, k+1)
        im = ax.imshow(
            surf_density.T[:,::-1],
            origin='lower',
            interpolation='nearest',
            vmin=0., vmax=vmax,
            cmap='magma',
            extent=extent
        )
        ax.set_xlabel(r'$x \ \left( \mathrm{kpc} \right)$')
        ax.set_ylabel(r'$y \ \left( \mathrm{kpc} \right)$')
        
        if len(it_label) >= k+1:
            label = it_label[k]
        else:
            label = r'$\mathrm{{iteration\ }} {}$'.format(i)
        
        xlim = ax.get_xlim()
        #x = xlim[0] + 0.97 * (xlim[1]-xlim[0])
        x = xlim[0] + 0.03
        ylim = ax.get_ylim()
        #y = ylim[0] + 0.03 * (ylim[1]-ylim[0])
        y = ylim[1] - 0.03
        ax.text(x, y, label,
                ha='left', va='top', fontsize=12,color='white')
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))

        ax.tick_params(which='both', left=True, right=True, bottom=True, top=True)
        
        if k % n_cols:
            if k % n_cols == n_cols-1:
                ax.tick_params(which='both', labelleft=False, labelright=True)
                ax.yaxis.set_label_position('right')
                right_label = True
            else:
                ax.tick_params(which='both', labelleft=False, labelright=False)
                ax.set_ylabel('')
        else:
            left_label = True

        if k >= n_rows*n_cols-n_cols: # Last row
            ax.tick_params(which='both', labelbottom=True, labeltop=False)
            ax.xaxis.set_label_position('bottom')
            bottom_label = True
        elif k < n_cols: # First row
            ax.tick_params(which='both', labelbottom=False, labeltop=True)
            ax.xaxis.set_label_position('top')
            top_label = True
        else:
            ax.tick_params(which='both', labelbottom=False, labeltop=False)
            ax.set_xlabel('')
    
    # Margins, spacing
    margins = dict(
        left=0.05, right=0.95,
        bottom=0.05, top=0.95,
        wspace=0.10, hspace=0.08
    )

    if bottom_label:
        margin_pct = 1.2 / h
        margins['bottom'] = margin_pct
    if top_label:
        margin_pct = 0.5 / h
        margins['top'] = 1. - margin_pct
    if left_label:
        margin_pct = 0.6 / w
        margins['left'] = margin_pct
    if right_label:
        margin_pct = 0.6 / w
        margins['right'] = 1. - margin_pct
    
    fig.subplots_adjust(**margins)
    
    # Colorbar
    cax = fig.add_axes([
        margins['left'],
        0.55/h,
        margins['right']-margins['left'],
        0.2/h
    ])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(r'$\mathrm{d}E / \mathrm{d}s \ \left( \mathrm{mag} \ \mathrm{kpc}^{-1} \right)$')
    
    it_suffix = '_it' + ','.join([str(i) for i in it])
    fn_base = './topdown/birdseye_'.format(it[-1])
    fn_base += 'surf_density' + suffix + it_suffix
    fig.savefig(fn_base + '.png', dpi=150)
    fig.savefig(fn_base + '.pdf', dpi=150)
    plt.close(fig)
    

def main():
    #for it in range(4):
    
    #it = 0

    #project_slices(it,20000.,300.)
    #plot_birdseye(it, 20000., 300., finite_frac=0.3, vmax=0.5)
    
    #project_slices(1,20000,300.)
    #project_slices(2,20000,300.)
    #project_slices(3,20000,300.)
    
    #project_slices(0,20000,300.)
    #project_slices(4,20000,300.)
    plot_birdseye([0,4], 20000., 300.,
                  finite_frac=0.0, vmax=0.4,
                  it_label=[r'$\mathrm{uncorrelated\ prior}$',
                            r'$\mathrm{correlated\ prior}$'])
    
    return 0


if __name__ == '__main__':
    main()
