import os, sys, argparse
sys.path.append("/n/fink2/czucker/BMK_Input/")

import matplotlib as mplib
mplib.use('pdf')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import h5py
import lsd
import pickle
from lsd import bounds
from lsd.builtins.misc import galequ, equgal
import sys
from astropy.table import Table
import shutil
import os
import iterators
import hputils
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from astropy import units as u


import os, sys, argparse

def tmass_hq_phot(obj):
    '''
    Return index of which detections in which bands
    are of high quality, as per the 2MASS recommendations:
    <http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_6b.html#composite>
    '''
    
    # Photometric quality in each passband
    idx = (obj['ph_qual'] == '0')
    obj['ph_qual'][idx] = '000'
    ph_qual = np.array(map(list, obj['ph_qual']))
    
    # Read flag in each passband
    idx = (obj['rd_flg'] == '0')
    obj['rd_flg'][idx] = '000'
    rd_flg = np.array(map(list, obj['rd_flg']))
    #rd_flg = (rd_flg == '1') | (rd_flg == '3')
    
    # Contamination flag in each passband
    idx = (obj['cc_flg'] == '0')
    obj['cc_flg'][idx] = '000'
    cc_flg = np.array(map(list, obj['cc_flg']))
    
    # Combine passband flags
    cond_1 = (ph_qual == 'A') | (rd_flg == '1') | (rd_flg == '3')
    cond_1 &= (cc_flg == '0')
    
    # Source quality flags
    cond_2 = (obj['use_src'] == 1) & (obj['gal_contam'] == 0)# & (obj['ext_key'] <= 0)
    
    # Combine all flags for each object
    hq = np.empty((len(obj), 3), dtype=np.bool)
    
    for i in range(3):
        hq[:,i] = cond_1[:,i] & cond_2
    
    return hq


def pix2lb(nside, ipix, nest=True):
    theta, phi = hp.pixelfunc.pix2ang(nside, ipix, nest=True)
    
    l = 180./np.pi * phi
    b = 90. - 180./np.pi * theta
    
    return l, b

def mapper(qresult, nside, nest, bounds):
    obj = lsd.colgroup.fromiter(qresult, blocks=True)
    
    if (obj != None) and (len(obj) > 0):
        # Determine healpix index of each star
        theta = np.pi/180. * (90. - obj['b'])
        phi = np.pi/180. * obj['l']
        pix_indices = hp.ang2pix(nside, theta, phi, nest=nest)
        
        # Group together stars having same index
        for pix_index, block_indices in iterators.index_by_key(pix_indices):
            yield (pix_index, obj[block_indices])
            
            
def reducer(keyvalue):
    pix_index, obj = keyvalue
    obj = lsd.colgroup.fromiter(obj, blocks=True)
    data=clean_data(obj)
    yield (pix_index, data)

def query_lsd(querystr, db=None, bounds=None, **kw):
    import lsd
    from lsd import DB, bounds as lsdbounds
    if db is None:
        db = os.environ['LSD_DB']
    if not isinstance(db, DB):
        dbob = DB(db)
    else:
        dbob = db
    if bounds is not None:
        bounds = lsdbounds.make_canonical(bounds)
    query = dbob.query(querystr, **kw)
    return query.fetch(bounds=bounds)
    
    
def clean_data(ob):


    good_det = np.empty((len(ob), 13), dtype=np.bool)
    
    good_det[:,:5] = (ob['decam_nmag_ok'] > 0) & (ob['decam_err'] < 0.2) & (ob['decam_fracflux'] > 0.85) & (np.isfinite(ob['decam_mean'])) & (np.isfinite(ob['decam_err'])) & (ob['decam_err'] > 0) & (ob['decam_mean'] < 50)
    
    
    good_det[:,5] = (ob['vvvz_Jerr'] < 0.2) & (ob['vvv_flagj'] == 0) & (np.isfinite(ob['vvvz_Jerr'])) & (np.isfinite(ob['vvvz_J'])) & (ob['vvvz_Jerr'] > 0.0) & (ob['vvvz_J'] < 50) & (ob['vvvz_J'] > 0) & (ob['vvv_flagkj']==0) & (ob['vvv_flaghj']==0) & (ob['vvv_flagjh']==0) & (ob['vvv_flagjk']==0)
    good_det[:,6] = (ob['vvvz_Herr'] < 0.2) & (ob['vvv_flagh'] == 0) & (np.isfinite(ob['vvvz_Herr'])) & (np.isfinite(ob['vvvz_H'])) & (ob['vvvz_Herr'] > 0.0) & (ob['vvvz_H'] < 50) & (ob['vvvz_H'] > 0) & (ob['vvv_flagkh']==0) & (ob['vvv_flaghj']==0) & (ob['vvv_flaghk']==0) & (ob['vvv_flagjh']==0)
    good_det[:,7] = (ob['vvvz_Kerr'] < 0.2) & (ob['vvv_flagk'] == 0) & (np.isfinite(ob['vvvz_Kerr'])) & (np.isfinite(ob['vvvz_K'])) & (ob['vvvz_Kerr'] > 0.0) & (ob['vvvz_K'] < 50) & (ob['vvvz_K'] > 0) & (ob['vvv_flagkj']==0) & (ob['vvv_flagkh']==0) & (ob['vvv_flaghk']==0) & (ob['vvv_flagjk']==0)

    hq_idx = tmass_hq_phot(ob)
    good_det[:,8:11] = hq_idx
    
    good_det[:,8] = good_det[:,8] & (ob['J_sig'] < 0.2)
    good_det[:,9] = good_det[:,9] & (ob['H_sig'] < 0.2)
    good_det[:,10] = good_det[:,10] & (ob['K_sig'] < 0.2)
    
    good_det[:,11] = (ob['W1_err']< 0.2) & (ob['W1_flags']==0) & (ob['W1_fracflux'] > 0.85) & (ob['W1'] < 50.) & (ob['W1'] > 0) & (ob['unwise_obj_primary.primary']==1)
    good_det[:,12] = (ob['W2_err'] < 0.2) & (ob['W2_flags']==0) & (ob['W2_fracflux'] > 0.85) & (ob['W2'] < 50.) & (ob['W2'] > 0) & (ob['unwise_obj_primary.primary']==1)
    
    #vvv_faint_blue_stars = (good_det[:,5]==1) & (good_det[:,6]==1) & (ob['vvvz_J'] - (0.705/(0.705-0.441))*(ob['vvvz_J']-ob['vvvz_H'] - 0.58) > 18)
    
    #either 4 detections with DECaPS-VVV or 4 with DECaPS-2MASS
    idx_good = ((np.sum(good_det[:,[0,1,2,3,4,5,6,7,11,12]], axis=1)>=4) | (np.sum(good_det[:,[0,1,2,3,4,8,9,10,11,12]], axis=1)>=4)) #& (~vvv_faint_blue_stars)

    # Copy in magnitudes and errors
    data = np.empty(len(ob), dtype=[('obj_id','u8'),
                                     ('l','f8'), ('b','f8'), ('ra','f8'), ('dec','f8'),
                                     ('decam_vvv_tmass_unwise_mag','13f4'), ('decam_vvv_tmass_unwise_err','13f4'),
                                     ('gaia_source_id','u8'),
                                     ('parallax','f4'),
                                     ('parallax_error','f4'),
                                     ('pmra','f4'),
                                     ('pmra_error','f4'),
                                     ('pmdec','f4'),
                                     ('pmdec_error','f4'),
                                     ('phot_g_mean_mag','f4'),
                                     ('phot_g_mean_mag_error','f4'), 
                                     ('phot_bp_mean_mag','f4'),
                                     ('phot_bp_mean_mag_error','f4'),
                                     ('phot_rp_mean_mag','f4'),
                                     ('phot_rp_mean_mag_error','f4'),
                                     ('nu_eff_used_in_astrometry','f4'),
                                     ('pseudocolour','f4'),
                                     ('ecl_lat','f4'),
                                     ('astrometric_params_solved','i8'),
                                     ('decam_fracflux','5f4'),     
                                     ('EBV','f4')])
                                     
                    
                                                        
    #DECaPS
    data['decam_vvv_tmass_unwise_mag'][:,:5] = ob['decam_mean'][:,:]
    data['decam_vvv_tmass_unwise_err'][:,:5] = ob['decam_err'][:,:]
    data['decam_fracflux'] = ob['decam_fracflux']

    
    #VVV
    data['decam_vvv_tmass_unwise_mag'][:,5] = ob['vvvz_J'][:]
    data['decam_vvv_tmass_unwise_err'][:,5] = ob['vvvz_Jerr'][:]
    data['decam_vvv_tmass_unwise_mag'][:,6] = ob['vvvz_H'][:]
    data['decam_vvv_tmass_unwise_err'][:,6] = ob['vvvz_Herr'][:]
    data['decam_vvv_tmass_unwise_mag'][:,7] = ob['vvvz_K'][:]
    data['decam_vvv_tmass_unwise_err'][:,7] = ob['vvvz_Kerr'][:]
    
    #2MASS
    data['decam_vvv_tmass_unwise_mag'][:,8] = ob['J'][:]
    data['decam_vvv_tmass_unwise_err'][:,8] = ob['J_sig'][:]
    data['decam_vvv_tmass_unwise_mag'][:,9] = ob['H'][:]
    data['decam_vvv_tmass_unwise_err'][:,9] = ob['H_sig'][:]
    data['decam_vvv_tmass_unwise_mag'][:,10] = ob['K'][:]
    data['decam_vvv_tmass_unwise_err'][:,10] = ob['K_sig'][:]
    
    data['decam_vvv_tmass_unwise_mag'][:,11] = ob['W1'][:]
    data['decam_vvv_tmass_unwise_err'][:,11] = ob['W1_err'][:]
    
    data['decam_vvv_tmass_unwise_mag'][:,12] = ob['W2'][:]
    data['decam_vvv_tmass_unwise_err'][:,12] = ob['W2_err'][:]
    
    
    #this is how eddie handles bad data
    data['decam_vvv_tmass_unwise_mag'][~good_det] = np.nan
    data['decam_vvv_tmass_unwise_err'][~good_det] = np.inf
    
    data['obj_id'][:] = ob['obj_id'][:]
    data['ra'][:] = ob['ra'][:]
    data['dec'][:] = ob['dec'][:]
    data['l'][:] = ob['l'][:]
    data['b'][:] = ob['b'][:]
    data['gaia_source_id'][:] = ob['source_id'][:]
    data['parallax'][:]=ob['parallax'][:]    
    data['parallax_error'][:]=ob['parallax_error']
    data['pmra'][:]=ob['pmra'][:]
    data['pmra_error'][:]=ob['pmra_error'][:]
    data['pmdec'][:]=ob['pmdec'][:]
    data['pmdec_error'][:]=ob['pmdec_error'][:]
    data['phot_g_mean_mag'][:]=ob['phot_g_mean_mag'][:]
    data['phot_g_mean_mag_error'][:]=ob['phot_g_mean_mag_error'][:]
    data['phot_bp_mean_mag'][:]=ob['phot_bp_mean_mag'][:]
    data['phot_bp_mean_mag_error'][:]=ob['phot_bp_mean_mag_error'][:]
    data['phot_rp_mean_mag'][:]=ob['phot_rp_mean_mag'][:]
    data['phot_rp_mean_mag_error'][:]=ob['phot_rp_mean_mag_error'][:]
    data['nu_eff_used_in_astrometry'][:] = ob['nu_eff_used_in_astrometry'][:]
    data['pseudocolour'][:] = ob['pseudocolour'][:]
    data['ecl_lat'][:] = ob['ecl_lat'][:]
    data['astrometric_params_solved'][:] = ob['astrometric_params_solved'][:]
   
    #restrict to stars with gaia parallaxes of reasonable quality
    good_parallax=(np.isfinite(ob['parallax']) & (ob['ruwe'] <= 1.4)) & (ob['parallax']!=0)
    data['parallax'][~good_parallax] = np.nan
    data['parallax_error'][~good_parallax] = np.inf
    
    data['pmra'][~good_parallax] = np.nan
    data['pmra_error'][~good_parallax] = np.inf
    
    data['pmdec'][~good_parallax] = np.nan
    data['pmdec_error'][~good_parallax] = np.inf

    data = data[idx_good]
    
    return data
    
    
def to_file(f, pix_index, nside, nest, EBV, data):

    close_file = False
    
    if type(f) == str:
        f = h5py.File(fname, 'a')
        close_file = True
    
    ds_name = '/photometry/pixel %d-%d' % (nside, pix_index)
    #ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=True,
    #                      compression='gzip', compression_opts=9)
    
    ds = f.create_dataset(ds_name, data.shape, data.dtype, chunks=(100,))
    
    ds[:] = data[:]
    
    N_stars = data.shape[0]
    t,p = hp.pixelfunc.pix2ang(nside, pix_index, nest=nest)
    t *= 180. / np.pi
    p *= 180. / np.pi
    gal_lb = np.array([p, 90. - t], dtype='f8')
    
    ds.attrs['healpix_index'] = pix_index
    ds.attrs['nested'] = nest
    ds.attrs['nside'] = nside
    ds.attrs['N_stars'] = N_stars
    ds.attrs['l'] = gal_lb[0]
    ds.attrs['b'] = gal_lb[1]
    ds.attrs['EBV'] = EBV
    
    if close_file:
        f.close()
    
    return gal_lb
    
    
if __name__ == '__main__':

    querystr="""select ra, dec, equgal(ra, dec) as (l, b), obj_id, SFD.EBV(l, b) as EBV, 2.5/log(10.)*np.array(decam_flux.err[:,1:6])/np.array(decam_flux.mean[:,1:6]) as decam_err, -2.5*log10(np.array(decam_flux.mean[:,1:6])) as decam_mean, np.array(decam_flux.fracflux[:,1:6]) as decam_fracflux, np.array(decam_flux.nmag_ok[:,1:6]) as decam_nmag_ok, gaia_edr3.source_id as source_id, gaia_edr3.ruwe as ruwe, gaia_edr3.parallax as parallax, gaia_edr3.parallax_error as parallax_error, gaia_edr3.pmra as pmra, gaia_edr3.pmra_error as pmra_error, gaia_edr3.pmdec as pmdec, gaia_edr3.pmdec_error as pmdec_error, gaia_edr3.phot_g_mean_mag as phot_g_mean_mag, 1.0857*gaia_edr3.phot_g_mean_flux/gaia_edr3.phot_g_mean_flux_error as phot_g_mean_mag_error, gaia_edr3.phot_bp_mean_mag as phot_bp_mean_mag, 1.0857*gaia_edr3.phot_bp_mean_flux/gaia_edr3.phot_bp_mean_flux_error as phot_bp_mean_mag_error, gaia_edr3.phot_rp_mean_mag as phot_rp_mean_mag, 1.0857*gaia_edr3.phot_rp_mean_flux/gaia_edr3.phot_rp_mean_flux_error as phot_rp_mean_mag_error, gaia_edr3.nu_eff_used_in_astrometry as nu_eff_used_in_astrometry, gaia_edr3.pseudocolour as pseudocolour, gaia_edr3.ecl_lat as ecl_lat, gaia_edr3.astrometric_params_solved as astrometric_params_solved, vvv_zhang.jmag_vista as vvvz_J, vvv_zhang.jerr as vvvz_Jerr, vvv_zhang.hmag_vista as vvvz_H, vvv_zhang.herr as vvvz_Herr, vvv_zhang.kmag_vista as vvvz_K, vvv_zhang.kerr as vvvz_Kerr, vvv_zhang.flag_j as vvv_flagj, vvv_zhang.flag_h as vvv_flagh, vvv_zhang.flag_k as vvv_flagk, vvv_zhang.flag_kj as vvv_flagkj, vvv_zhang.flag_kh as vvv_flagkh, vvv_zhang.flag_hj as vvv_flaghj, vvv_zhang.flag_hk as vvv_flaghk, vvv_zhang.flag_jh as vvv_flagjh, vvv_zhang.flag_jk as vvv_flagjk, tmass.designation, tmass.ph_qual as ph_qual, tmass.use_src as use_src, tmass.rd_flg as rd_flg, tmass.ext_key as ext_key, tmass.gal_contam as gal_contam, tmass.cc_flg as cc_flg,tmass.j_m as J, tmass.j_msigcom as J_sig, tmass.h_m as H, tmass.h_msigcom as H_sig, tmass.k_m as K, tmass.k_msigcom as K_sig, unwise_obj_primary.unwise_objid, 22.5-2.5*log10(clip(unwise_obj_primary.flux(0), 1e-30, inf)) as W1, 22.5-2.5*log10(clip(unwise_obj_primary.flux(1), 1e-30, inf)) as W2, 1.0857*(unwise_obj_primary.dflux(0)/clip(unwise_obj_primary.flux(0),1e-12,inf)) as W1_err, 1.0857*(unwise_obj_primary.dflux(1)/clip(unwise_obj_primary.flux(1),1e-12,inf)) as W2_err, unwise_obj_primary.fracflux(0) as W1_fracflux, unwise_obj_primary.fracflux(1) as W2_fracflux, unwise_obj_primary.flags_unwise(0) as W1_flags, unwise_obj_primary.flags_unwise(1) as W2_flags, unwise_obj_primary.primary, unwise_obj_primary.flags_info from decam_flux, vvv_zhang(outer, matchedto=decam_flux, dmax=0.5, nmax=1), gaia_edr3(outer,matchedto=decam_flux, dmax=0.5, nmax=1),  tmass(outer,matchedto=decam_flux, dmax=0.5, nmax=1), unwise_obj_primary(outer,matchedto=decam_flux, dmax=0.5, nmax=1)"""
                                     
                                                   
    # Set up the query
    os.environ['LSD_DB']='/n/holylfs05/LABS/finkbeiner_lab/Everyone/lsd_db/'

    db = lsd.DB(os.environ['LSD_DB'])

    sfd = SFDQuery()
        
    # Write each pixel to the same file
    nest = True
    nside=1024
    

    #l_min, l_max, b_min, b_max = 331, 334, 0.5, 3.5
    l_min, l_max, b_min, b_max = 308, 320, -2, 2
    
    value_bounds = [l_min, l_max, b_min, b_max]
    
    # Determine the query bounds
    pix_scale = hp.pixelfunc.nside2resol(nside) * 180. / np.pi
    query_bounds = []
    query_bounds.append(value_bounds[0] - 1.*pix_scale) 
    query_bounds.append(value_bounds[1] + 1.*pix_scale)
    query_bounds.append(max([-90., value_bounds[2] - 1.*pix_scale]))
    query_bounds.append(min([90., value_bounds[3] + 1.*pix_scale]))
    
    
    query_bounds = lsd.bounds.rectangle(query_bounds[0], query_bounds[2],
                                            query_bounds[1], query_bounds[3],
                                            coordsys='gal')
                                            

    query_bounds = None
    query_bounds = lsd.bounds.make_canonical(query_bounds)
    query = db.query(querystr)

    f = None
    nFiles = 0
    nInFile = 0
    N_stars = 0
    max_stars = 10000
    fnameBase =  '/n/fink2/czucker/Plane_Final/full_plane/'


    fnameSuffix = 'h5'
    stars_in_healpix_arr = [] 
    healpix_arr = []

    for (pix_index, ob) in query.execute([(mapper, nside, nest, value_bounds),reducer],bounds=query_bounds, nworkers=6):  
            
        l_center, b_center = pix2lb(nside, pix_index, nest=nest)
        
        if not hputils.lb_in_bounds(l_center, b_center, value_bounds):
            continue
            
        # Open output file
        if f == None:
            fname = '%s.%.5d.%s' % (fnameBase, nFiles, fnameSuffix)
            f = h5py.File(fname, 'w')
            nInFile = 0
            nFiles += 1
            
            
        coords = SkyCoord(l=ob['l']*u.deg, b=ob['b']*u.deg, frame='galactic')
        ebv_stars = sfd(coords)
        
        # Prepare output for pixel
        EBV = np.nanpercentile(ebv_stars, 95.)
        
        # Write to file
        gal_lb = to_file(f, pix_index, nside, nest, EBV, ob)
        
        stars_in_pix =len(ob)
        N_stars+= stars_in_pix
        nInFile+= stars_in_pix
        
        stars_in_healpix_arr.append(stars_in_pix)
        healpix_arr.append(pix_index)

        # Close file if size exceeds max_stars
        if nInFile >= max_stars:
            f.close()
            f = None
    
    if f != None:
        f.close()
        
    np.save('stars_in_healpix.npy',np.array(stars_in_healpix_arr))
    np.save('healpix_index.npy',np.array(healpix_arr))

    print('nstars',N_stars)
