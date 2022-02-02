import os
from brutus import filters
import h5py
import numpy as np
import brutus
import sys
import glob
from brutus import utils
from brutus.utils import inv_magnitude
from brutus import fitting
from scipy import stats
from brutus import pdf
import signal
import time
from zero_point import zpt
import shutil
from memory_profiler import memory_usage


global f_out

filename_index = int(sys.argv[1])

filenames = glob.glob('/n/fink2/czucker/Plane_Final/input/G314/G314*.h5')

filenames = np.sort(filenames)
filename = filenames[filename_index]

output_filename = '/n/fink2/czucker/Plane_Final/perstar/G314/{}'.format(filename.split('/')[-1][:-3])


#load filters
decam_vvv_tmass_unwise_filt = filters.decam[1:] + filters.vista[2:] + filters.tmass + filters.wise[0:2]

#load models
(models_mist, labels_mist,
 lmask_mist) = utils.load_models('{}/grid_mist_v9.h5'.format(os.environ['work_dir']), filters=decam_vvv_tmass_unwise_filt, include_ms=True, 
                                  include_postms=True, include_binaries=False)


BF_mist = fitting.BruteForce(models_mist, labels_mist, lmask_mist)

def handle_signal(signal_value, _):

    signame = signal.Signals(signal_value).name
    print('Process {} got signal {}.'.format(os.getpid(), signame), flush=True)
    f_out.close()

    time.sleep(120)

signal.signal(signal.SIGUSR1, handle_signal)
signal.signal(signal.SIGCONT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGQUIT, handle_signal)
signal.signal(signal.SIGILL, handle_signal)
signal.signal(signal.SIGTRAP, handle_signal)
signal.signal(signal.SIGABRT, handle_signal)
signal.signal(signal.SIGBUS, handle_signal)
signal.signal(signal.SIGFPE, handle_signal)
signal.signal(signal.SIGSEGV, handle_signal)
signal.signal(signal.SIGUSR2, handle_signal)
signal.signal(signal.SIGPIPE, handle_signal)
signal.signal(signal.SIGALRM, handle_signal)


f_in = h5py.File(filename,'r',swmr=True)

datasets = [key for key in f_in['photometry'].keys()]

#load zeropoints
zp_mist = utils.load_offsets('{}/offsets_mist_v9.txt'.format(os.environ['work_dir']), filters=decam_vvv_tmass_unwise_filt)


#load tables for parallax zeropoint correction
zpt.load_tables()

try:
    f_out = h5py.File(output_filename+".h5",'a')

except:

    try:
        print("first try")
        f_out = h5py.File(output_filename+".h5",'r+')

    except:
        print('second try')
        os.remove(output_filename+".h5")
        f_out = h5py.File(output_filename+".h5",'a')


#run brutus fitting 
for dataset in datasets: 

    fpix = f_in['photometry/{}'.format(dataset)]

    Nobjs = f_in['photometry/{}'.format(dataset)].attrs['N_stars']
    Ndraws = 250

    if '/stellar_samples/{}/labels'.format(dataset) not in f_out:

        f_out.create_dataset("/stellar_samples/{}/labels".format(dataset),data=np.full((Nobjs),0, dtype='int32'),chunks=(100,))
        f_out.create_dataset("/stellar_samples/{}/model_idx".format(dataset), data=np.full((Nobjs, Ndraws), -99, dtype='int32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/ml_scale".format(dataset), data=np.ones((Nobjs, Ndraws), dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/ml_av".format(dataset), data=np.zeros((Nobjs, Ndraws),dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/ml_rv".format(dataset), data=np.zeros((Nobjs, Ndraws),dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/ml_cov_sar".format(dataset), data=np.zeros((Nobjs, Ndraws, 3, 3), dtype='float32'),chunks=(100,250,3,3))
        f_out.create_dataset("/stellar_samples/{}/obj_log_post".format(dataset), data=np.zeros((Nobjs, Ndraws), dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/obj_log_evid".format(dataset), data=np.zeros(Nobjs, dtype='float32'),chunks=(100,))
        f_out.create_dataset("/stellar_samples/{}/obj_chi2min".format(dataset), data=np.zeros(Nobjs, dtype='float32'),chunks=(100,))
        f_out.create_dataset("/stellar_samples/{}/obj_Nbands".format(dataset), data=np.zeros(Nobjs, dtype='int16'),chunks=(100,))
        f_out.create_dataset("/stellar_samples/{}/samps_dist".format(dataset), data=np.ones((Nobjs, Ndraws),dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/samps_red".format(dataset), data=np.ones((Nobjs, Ndraws), dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/samps_dred".format(dataset), data=np.ones((Nobjs, Ndraws), dtype='float32'),chunks=(100,250))
        f_out.create_dataset("/stellar_samples/{}/samps_logp".format(dataset), data=np.ones((Nobjs, Ndraws), dtype='float32'),chunks=(100,250))

        f_out['stellar_samples/{}'.format(dataset)].attrs['batch_num'] = 0
        batch_num = f_out['stellar_samples/{}'.format(dataset)].attrs['batch_num']

    elif f_out['stellar_samples/{}'.format(dataset)].attrs['batch_num'] > len(list(fpix.iter_chunks())):

        continue
    else:
        batch_num = f_out['stellar_samples/{}'.format(dataset)].attrs['batch_num']

    # fit a set of hypothetical objects
    # MIST

    chunks = range(batch_num,len(list(fpix.iter_chunks())))
    batches = list(fpix.iter_chunks())[batch_num:]
    batches_1d = list(f_out['/stellar_samples/{}/obj_Nbands'.format(dataset)].iter_chunks())[batch_num:]
    batches_2d = list(f_out['/stellar_samples/{}/samps_dist'.format(dataset)].iter_chunks())[batch_num:]
    batches_covs = list(f_out['/stellar_samples/{}/ml_cov_sar'.format(dataset)].iter_chunks())[batch_num:]

    for (batch, chunk_index, batch_1d, batch_2d, batch_cov)  in zip(batches, chunks,batches_1d,batches_2d, batches_covs):        

        # # DECam-VVV-2MASS MIST
        # load in fitter
        mag, magerr = fpix['decam_vvv_tmass_unwise_mag'][batch], fpix['decam_vvv_tmass_unwise_err'][batch]

        #add systematic error corrections

        #add 0.02 mag uncertainty in quadrature to decaps
        magerr[:,0:5] = np.sqrt(magerr[:,0:5]**2 + 0.02**2)


        #add 0.03 mag uncertainty in quadrature to vvv/2mass
        magerr[:,5:] = np.sqrt(magerr[:,5:]**2 + 0.03**2)

        mask = np.isfinite(magerr)  # create boolean band mask
        phot, err = inv_magnitude(mag, magerr)  # convert to flux
        objid = fpix['obj_id'][batch]
        parallax, parallax_err = fpix['parallax'][batch], fpix['parallax_error'][batch]
        coords = np.c_[fpix['l'][batch], fpix['b'][batch]]

        #apply parallax correction
        correct_parallax_mask = np.isfinite(parallax)
        parallax_correction = zpt.get_zpt(fpix['phot_g_mean_mag'][batch][correct_parallax_mask], fpix['nu_eff_used_in_astrometry'][batch][correct_parallax_mask], fpix['pseudocolour'][batch][correct_parallax_mask], fpix['ecl_lat'][batch][correct_parallax_mask], fpix['astrometric_params_solved'][batch][correct_parallax_mask],_warnings=True)
        parallax_correction[~np.isfinite(parallax_correction)] = 0 
        parallax[correct_parallax_mask] = parallax[correct_parallax_mask]-parallax_correction

        #add in systematic error to parallaxes

        f_out['/stellar_samples/{}/labels'.format(dataset)][batch_1d] = objid

        BF_mist.fit(phot, err, mask, objid, 
                    f_out, batch_1d, batch_2d, batch_cov, dataset,
                    parallax=parallax, parallax_err=parallax_err, 
                    data_coords=coords, 
                    avlim=(0,24),
                    av_gauss=(12,1e+6),
                    phot_offsets=zp_mist,
                    Ndraws=Ndraws,  # number of samples to save
                    Nmc_prior=100,  # number of Monte Carlo draws used to integrate priors
                    logl_dim_prior=True,  # use chi2 distribution instead of Gaussian
                    save_dar_draws=True,  # save (dist, Av, Rv) samples
                    running_io=False, mem_lim=4000)

        f_out['stellar_samples/{}'.format(dataset)].attrs['batch_num'] = chunk_index + 1

#regrid to bayestar PDFS 
for dataset in datasets: 

    dists = f_out['/stellar_samples/{}/samps_dist'.format(dataset)][:]  # distance samples
    reds = f_out['/stellar_samples/{}/samps_red'.format(dataset)][:]  # A(V) samples
    dreds = f_out['/stellar_samples/{}/samps_dred'.format(dataset)][:]  # R(V) samples
    chi2 = f_out['/stellar_samples/{}/obj_chi2min'.format(dataset)][:]  # best-fit chi2
    nbands = f_out['/stellar_samples/{}/obj_Nbands'.format(dataset)][:]  # number of bands in fit
    idxs = f_out['/stellar_samples/{}/model_idx'.format(dataset)][:]  # model indices
    fpix = f_in['photometry/{}'.format(dataset)]
    parallax, parallax_err = fpix['parallax'], fpix['parallax_error']
    coords = np.c_[fpix['l'], fpix['b']]

    gridmask = np.percentile(labels_mist['mini'][idxs], 2.5, axis=1) > 0.5
    nearmask = np.percentile(dists, 50, axis=1) < 30

    good=(stats.chi2.sf(chi2, nbands) > 0.01) & gridmask & nearmask

    pdfbin, xedges, yedges = pdf.bin_pdfs_distred((dists[good],reds[good],dreds[good]), parallaxes=parallax[good],parallax_errors=parallax_err[good],coord=coords[good],ebv=True,bins=(120,700),avlim=(0,7),smooth=0.01)    
    pdfbin=np.transpose(pdfbin,axes=(0,2,1))
    #normalize pdfs
    pdf_sum = np.sum(pdfbin,axis=(1,2))
    pdfbin = pdfbin / pdf_sum[:, np.newaxis,np.newaxis]     

    #remove stray nans
    pdfmask = np.isfinite(np.sum(pdfbin,axis=(1,2)))
    pdfbin=pdfbin[pdfmask,:,:]

    dname='/stellar_pdfs/%s/stellar_pdfs'%(dataset)
    dname2='/stellar_pdfs/%s'%(dataset)

    if dname2 in f_out:
        del f_out[dname2]
    dset = f_out.create_dataset(
            dname,
            data=pdfbin,
            chunks=True,
            compression='gzip',
            compression_opts=3
        )

    # Add attributes
    dset.attrs['nPix'] = np.array(list(dset.shape[1:]), dtype='u4')
    dset.attrs['min'] = np.array([0.,4.], dtype='f8')
    dset.attrs['max'] = np.array([7.,19.], dtype='f8')
    g = f_out[dname2]
    g.attrs['l'] = f_in['photometry/{}'.format(dataset)].attrs['l']
    g.attrs['b'] = f_in['photometry/{}'.format(dataset)].attrs['b']
    g.attrs['nside'] = f_in['photometry/{}'.format(dataset)].attrs['nside']
    g.attrs['healpix_index'] = f_in['photometry/{}'.format(dataset)].attrs['healpix_index']
    g.attrs['EBV'] = f_in['photometry/{}'.format(dataset)].attrs['EBV']
    dname_mask='/stellar_mask/%s/stellar_mask'%(dataset)

    if dname_mask in f_out:
        del f_out[dname_mask]

    dset_mask = f_out.create_dataset(
            dname_mask,
            data=good,
            chunks=True,
            compression='gzip',
            compression_opts=3
        )

f_out.close()
f_in.close()

