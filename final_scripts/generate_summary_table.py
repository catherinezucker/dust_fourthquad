from scipy import stats
from brutus.utils import inv_magnitude
import sys
import h5py
import numpy as np
from astropy.table import Table
from brutus import filters
from brutus import utils
import glob

#load filters
decam_vvv_tmass_filt = filters.decam[1:] + filters.vista[2:] + filters.tmass + filters.wise[0:2]


#load models
(models_mist, labels_mist,
 lmask_mist) = utils.load_models('/n/fink2/czucker/Plane_Final/models_and_offsets/grid_mist_v9.h5', filters=decam_vvv_tmass_filt, include_ms=True, 
                                  include_postms=True, include_binaries=False)

filename_index = int(sys.argv[1])


filename = '/n/fink2/czucker/Plane_Final/input/G314/G314.{:05d}.h5'.format(filename_index)

output_filename = '/n/fink2/czucker/Plane_Final/perstar/G314/G314.{:05d}.h5'.format(filename_index)

f_in = h5py.File(filename,'r')
f_out = h5py.File(output_filename,'r')

datasets = [key for key in f_in['photometry'].keys()]


#run brutus fitting 
for dataset in datasets: 

    fpix = f_in['photometry/{}'.format(dataset)]
    mag, magerr = fpix['decam_vvv_tmass_unwise_mag'], fpix['decam_vvv_tmass_unwise_err']
    mask = np.isfinite(magerr)  # create boolean band mask
    phot, err = inv_magnitude(mag, magerr)  # convert to flux
    objid = fpix['obj_id']
    parallax, parallax_err = fpix['parallax'], fpix['parallax_error']
    coords = np.c_[fpix['l'], fpix['b']]
 
    dists = f_out['/stellar_samples/{}/samps_dist'.format(dataset)][:]  # distance samples
    reds = f_out['/stellar_samples/{}/samps_red'.format(dataset)][:]  # A(V) samples
    dreds = f_out['/stellar_samples/{}/samps_dred'.format(dataset)][:]  # R(V) samples
    chi2 = f_out['/stellar_samples/{}/obj_chi2min'.format(dataset)][:]  # best-fit chi2
    nbands = f_out['/stellar_samples/{}/obj_Nbands'.format(dataset)][:]  # number of bands in fit
    idxs = f_out['/stellar_samples/{}/model_idx'.format(dataset)][:]  # model indices
        
    gridmask=np.percentile(labels_mist['mini'][idxs], 2.5, axis=1) > 0.5
    good=((stats.chi2.sf(chi2, nbands) > 0.01) & gridmask)
    
    t=Table()
    t['mask']=good
    
    good = np.ones((len(gridmask))).astype(bool)

    t['l']=fpix['l'][:][good]
    t['b']=fpix['b'][:][good]
    t['ra']=fpix['ra'][:][good]
    t['dec']=fpix['dec'][:][good]
    t['parallax']=fpix['parallax'][good]
    t['parallax_error']=fpix['parallax_error'][good]
    t['g']=fpix['decam_vvv_tmass_unwise_mag'][:,0][good]
    t['r']=fpix['decam_vvv_tmass_unwise_mag'][:,1][good]
    t['i']=fpix['decam_vvv_tmass_unwise_mag'][:,2][good]
    t['z']=fpix['decam_vvv_tmass_unwise_mag'][:,3][good]
    t['y']=fpix['decam_vvv_tmass_unwise_mag'][:,4][good]
    t['J_vvv']=fpix['decam_vvv_tmass_unwise_mag'][:,5][good]
    t['H_vvv']=fpix['decam_vvv_tmass_unwise_mag'][:,6][good]
    t['K_vvv']=fpix['decam_vvv_tmass_unwise_mag'][:,7][good]
    t['J_tm']=fpix['decam_vvv_tmass_unwise_mag'][:,8][good]
    t['H_tm']=fpix['decam_vvv_tmass_unwise_mag'][:,9][good]
    t['K_tm']=fpix['decam_vvv_tmass_unwise_mag'][:,10][good]

    t['gerr']=fpix['decam_vvv_tmass_unwise_err'][:,0][good]
    t['rerr']=fpix['decam_vvv_tmass_unwise_err'][:,1][good]
    t['ierr']=fpix['decam_vvv_tmass_unwise_err'][:,2][good]
    t['zerr']=fpix['decam_vvv_tmass_unwise_err'][:,3][good]
    t['yerr']=fpix['decam_vvv_tmass_unwise_err'][:,4][good]
    t['Jerr_vvv']=fpix['decam_vvv_tmass_unwise_err'][:,5][good]
    t['Herr_vvv']=fpix['decam_vvv_tmass_unwise_err'][:,6][good]
    t['Kerr_vvv']=fpix['decam_vvv_tmass_unwise_err'][:,7][good]
    t['Jerr_tm']=fpix['decam_vvv_tmass_unwise_err'][:,8][good]
    t['Herr_tm']=fpix['decam_vvv_tmass_unwise_err'][:,9][good]
    t['Kerr_tm']=fpix['decam_vvv_tmass_unwise_err'][:,10][good]
    
    '''
    t['vvv_zhang.jsky'] = fpix['vvv_zhang.jsky'][:][good]
    t['vvv_zhang.hsky'] = fpix['vvv_zhang.hsky'][:][good]
    t['vvv_zhang.ksky'] = fpix['vvv_zhang.ksky'][:][good]
    t['vvv_zhang.jniter'] = fpix['vvv_zhang.jniter'][:][good]
    t['vvv_zhang.hniter'] = fpix['vvv_zhang.hniter'][:][good]
    t['vvv_zhang.kniter'] = fpix['vvv_zhang.kniter'][:][good]
    t['vvv_zhang.jsharpness'] = fpix['vvv_zhang.jsharpness'][:][good]
    t['vvv_zhang.hsharpness'] = fpix['vvv_zhang.hsharpness'][:][good]
    t['vvv_zhang.ksharpness'] = fpix['vvv_zhang.ksharpness'][:][good]
    t['vvv_zhang.jchi'] = fpix['vvv_zhang.jchi'][:][good]
    t['vvv_zhang.hchi'] = fpix['vvv_zhang.hchi'][:][good]
    t['vvv_zhang.kchi'] = fpix['vvv_zhang.kchi'][:][good]
    t['vvv_zhang.jpier'] = fpix['vvv_zhang.jpier'][:][good]
    t['vvv_zhang.hpier'] = fpix['vvv_zhang.hpier'][:][good]
    t['vvv_zhang.kpier'] = fpix['vvv_zhang.kpier'][:][good]
    '''
    dms =5. * np.log10(dists) + 10.

    t['dms']=np.median(dms,axis=1)[good]
    t['reds']=np.median(reds,axis=1)[good]
    t['dreds']=np.median(dreds,axis=1)[good]
    t['dists']=np.median(dists,axis=1)[good]
    
    props=['mini','feh','eep','loga','logl','logt','logg']

    for prop in props:
        t[prop+"_16"]=np.percentile(labels_mist[prop][idxs],16,axis=1)[good]
        t[prop+"_50"]=np.percentile(labels_mist[prop][idxs],50,axis=1)[good]
        t[prop+"_84"]=np.percentile(labels_mist[prop][idxs],84,axis=1)[good]
        
    t['nbands']=nbands[good]
    t['chi2']=chi2[good]
    
    t.write("/n/fink2/czucker/Plane_Final/diagnostics/diagnostic_{}.fits".format(dataset.replace(" ","_")),overwrite=True)

f_in.close()
f_out.close()
