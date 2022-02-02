#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py
from argparse import ArgumentParser
from progressbar import ProgressBar
from collections import Counter


def file_stats(fn, n_tau):
    with h5py.File(fn, 'r') as f:
        d = f['pix_info'][:]
    
    c = Counter({
        'n_pix': d.size,
        'n_conv': int(np.sum(d['n_tau'] >= n_tau)),
        #'t_star': np.sum(d['runtime_star']),
        't_los': np.sum(d['runtime_los']),
        #'f_reject': np.sum(d['runtime_star']*d['reject_frac'])
    })
    
    return c


def main():
    parser = ArgumentParser(
        description='Report fraction of pixels that pass convergence test.',
        add_help=True
    )
    parser.add_argument(
        'outfiles',
        metavar='*.h5',
        nargs='+',
        help='Combined output files.'
    )
    parser.add_argument(
        '-n', '--n-tau',
        metavar='TAU_MIN',
        default=20.,
        help='Cut-off in n_tau for converngence (default: 20).'
    )
    args = parser.parse_args()
    
    c = Counter()
    
    bar = ProgressBar(max_value=len(args.outfiles))
    bar.update(0)
    
    for k,fn in enumerate(args.outfiles):
        c += file_stats(fn, args.n_tau)
        bar.update(k+1)
    
    print('    n_pix = {:d}'.format(c['n_pix']))
    print('   n_conv = {:d}'.format(c['n_conv']))
    print('f_nonconv = {:.4g}'.format(1.-(c['n_conv']/c['n_pix'])))
    #print('        t = {:.1f} s'.format(c['t_star'] + c['t_los']))
    #print('   t_star = {:.1f} s'.format(c['t_star']))
    print('    t_los = {:.1f} s'.format(c['t_los']))
    #print(' f_reject = {:.4g}'.format(c['f_reject'] / c['t_star']))
    
    return 0


if __name__ == '__main__':
    main()
