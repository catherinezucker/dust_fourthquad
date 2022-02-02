#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py
from argparse import ArgumentParser
from glob import glob
from progressbar import ProgressBar

def PCA(data):
    # Calculate the eigendecomposition of the covariance matrix
    C = np.cov(data, rowvar=False)
    eival, eivec = np.linalg.eigh(C)
    
    # Normalize eigenvectors (unit length) and rescale eigenvalues
    norm = np.linalg.norm(eivec, axis=1)
    eivec = eivec[:,:] / norm[:,None]
    eival *= norm
    
    # Sort the eigenvalues/eigenvectors (largest to smallest)
    idx = np.argsort(eival)[::-1]
    eival = eival[idx]
    eivec = eivec[:,idx]
    
    # Transform the data to the new coordinate system
    d_transf = np.dot(data - np.mean(data, axis=0), eivec)

    # Returns the (eigenvalues, eigenvectors, transormed data)
    return eival, eivec, d_transf


def autocorr_1d(y, threshold=0.05):
    """
    Calculates the autocorrelation of a 1-dimensional
    signal, y.
    
    Inputs:
        y (array-like): 1-dimensional signal.
    
    Returns:
        Autocorrelation as a function of displacement,
        and an estimate of the autocorrelation time,
        based on the smallest displacement with a
        negative autocorrelation.

    From the StackOverflow answer by unutbu:
    <https://stackoverflow.com/a/14298647/1103939>
    """
    n = len(y)
    y0 = np.mean(y)
    sigma2 = np.var(y)
    dy = y - y0
    
    if sigma2 == 0:
        return np.ones(n), -1.0
    
    r = np.correlate(dy, dy, mode='full')[-n:]
    r /= (sigma2 * np.arange(n, 0, -1))
    
    idx = np.where(r < threshold)[0]
    
    if not len(idx):
        return r, -1.0
    
    tau = idx[0]
    return r, tau


def rel2abs_coords(ax, x, y):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    return (xlim[0] + x*w, ylim[0] + y*h)


def get_n_tau_min(dset):

    ln_like = dset[:,0]
    ln_prior = dset[:,1]
    theta = dset[:,2:]

    data = {'ln_like': ln_like, 'ln_prior': ln_prior, 'theta': theta}
    
    # Autocorrelation of log(prior) and log(likelihood)
    acorr_prior, tau_prior = autocorr_1d(
        data['ln_prior'],
        threshold=0.05)
    acorr_like, tau_like = autocorr_1d(
        data['ln_like'],
        threshold=0.05)
    
    # Autocorrelation of principal component coefficients
    eival, eivec, chain_transf = PCA(data['theta'])
    acorr, tau = autocorr_1d(
        chain_transf[:,0],
        threshold=0.05)
    
    tau_max = max([tau, tau_prior, tau_like])
    tau_max_idx = np.argmax([tau, tau_prior, tau_like])
    n_tau_min = acorr.size / tau_max

    return n_tau_min

def load_file(fname):
    with h5py.File(fname, 'r') as f:
        n_pix = len(f.keys())
        
        shape = f[[key for key in f.keys()][0]]['discrete-los'].shape[1:]


        shape = (n_pix,) + shape
        
        chains = np.empty(shape, dtype='f4')
        
        dtype = [
            ('nside', 'i4'),
            ('pix_idx', 'i4'),
            ('runtime_star', 'f4'),
            ('runtime_los', 'f4'),
            ('n_tau', 'f4'),
            ('reject_frac', 'f4')
        ]
        pix_info = np.empty(n_pix, dtype=dtype)
        
        for k,key in enumerate(f.keys()):
            # dset.shape = (chain, GR+best+sample, lnlike+lnprior+parameter)
            chains[k,:,:] = 0.01 * f[key]['discrete-los'][0,:,:]
            
            n,i = key.split()[1].split('-')
            pix_info[k]['nside'] = int(n)
            pix_info[k]['pix_idx'] = int(i)
            pix_info[k]['runtime_los'] = f[key]['discrete-los'].attrs['runtime']
            
            n_tau = get_n_tau_min(f[key]['discrete-los'][0,2:,:])
            pix_info[k]['n_tau'] = n_tau
    
    return chains, pix_info


def combine_files(in_fnames, out_fname):
    print('Loading bayestar output files ...')
    bar = ProgressBar(max_value=len(in_fnames))
    bar.update(0)
    
    chains, pix_info = [], []
    
    for k,fn in enumerate(in_fnames):
        c,p = load_file(fn)
        chains.append(c)
        pix_info.append(p)
        
        bar.update(k+1)
    
    print('Concatenating data ...')
    chains = np.concatenate(chains, axis=0)
    pix_info = np.hstack(pix_info)
    
    print('Writing combined output file ...')
    
    with h5py.File(out_fname, 'w') as f:
        f.create_dataset('discrete-los', data=chains, chunks=True, compression='gzip', compression_opts=3)
        f.create_dataset('pix_info', data=pix_info, chunks=True, compression='gzip', compression_opts=3)


def main():
    parser = ArgumentParser(description='Concatenate bayestar output files.',
                            add_help=True)
    parser.add_argument('-i', '--inputs', type=str, nargs='+',  help='Bayestar output files.')
    parser.add_argument('-o', '--output', type=str, help='Combined output filename.')
    args = parser.parse_args()
    
    fnames = []
    for fn in args.inputs:
        fnames += glob(fn)
    
    combine_files(fnames, args.output)
    
    return 0


if __name__ == '__main__':
    main()
