from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack
import fitsio

def get_required_columns(galaxy_type):
    if galaxy_type[:3]=="BGS":
        return ['TARGETID', 'PHOTSYS', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FIBERFLUX_R', 'FIBERTOTFLUX_R', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'SHAPE_R', 'TSNR2_BGS', 'ZWARN', 'DELTACHI2']
    elif galaxy_type[:3]=="LRG":
        return ['TARGETID', 'PHOTSYS', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FIBERFLUX_Z', 'FIBERTOTFLUX_Z', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'SHAPE_R', 'TSNR2_ELG', 'ZWARN', 'DELTACHI2']
    else:
        raise NotImplementedError("galaxy_type {} not implemented".format(galaxy_type))
    

def select_lrg(cat, field='south'):
    '''
    columns = ['OBJID', 'BRICKID', 'RELEASE', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FIBERFLUX_Z', 'FIBERTOTFLUX_Z', 'GAIA_PHOT_G_MEAN_MAG', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z']
    fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/sweep-000p000-010p005.fits'
    cat = Table(fitsio.read(fn, columns=columns))
    mask_lrg = select_lrg(cat)
    '''

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    mask_quality = np.full(len(cat), True)

    mask_quality &= (cat['FLUX_IVAR_R'] > 0) & (cat['FLUX_R'] > 0)   # ADM quality in r.
    mask_quality &= (cat['FLUX_IVAR_Z'] > 0) & (cat['FLUX_Z'] > 0) & (cat['FIBERFLUX_Z'] > 0)   # ADM quality in z.
    mask_quality &= (cat['FLUX_IVAR_W1'] > 0) & (cat['FLUX_W1'] > 0)  # ADM quality in W1.

    # mask_quality &= (cat['GAIA_PHOT_G_MEAN_MAG'] == 0) | (cat['GAIA_PHOT_G_MEAN_MAG'] > 18)  # remove bright GAIA sources

    # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
    mask_quality &= cat['FIBERTOTFLUX_Z'] < 10**(-0.4*(17.5-22.5))

    # ADM observed in every band.
    mask_quality &= (cat['NOBS_G'] > 0) & (cat['NOBS_R'] > 0) & (cat['NOBS_Z'] > 0)

    # Apply masks
    maskbits = [1, 12, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))
    mask_quality &= mask_clean

    # gmag = 22.5 - 2.5 * np.log10((cat['FLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))
    # # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    # rmag = 22.5 - 2.5 * np.log10((cat['FLUX_R'] / cat['MW_TRANSMISSION_R']).clip(1e-7))
    # zmag = 22.5 - 2.5 * np.log10((cat['FLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))
    # w1mag = 22.5 - 2.5 * np.log10((cat['FLUX_W1'] / cat['MW_TRANSMISSION_W1']).clip(1e-7))
    # zfibermag = 22.5 - 2.5 * np.log10((cat['FIBERFLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))
    gmag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_G']*10**(0.4*3.214*cat['EBV']), 1e-7, None))
    rmag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_R']*10**(0.4*2.165*cat['EBV']), 1e-7, None))
    zmag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_Z']*10**(0.4*1.211*cat['EBV']), 1e-7, None))
    w1mag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_W1']*10**(0.4*0.184*cat['EBV']), 1e-7, None))
    zfibermag = 22.5 - 2.5*np.log10(np.clip(cat['FIBERFLUX_Z']*10**(0.4*1.211*cat['EBV']), 1e-7, None))

    mask_lrg = mask_quality.copy()

    if field=='south':
        mask_lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        mask_lrg &= zfibermag < 21.6                   # faint limit
        mask_lrg &= (gmag - w1mag > 2.9) | (rmag - w1mag > 1.8)  # low-z cuts
        mask_lrg &= (
            ((rmag - w1mag > (w1mag - 17.14) * 1.8)
             & (rmag - w1mag > (w1mag - 16.33) * 1.))
            | (rmag - w1mag > 3.3)
        )  # double sliding cuts and high-z extension
    else:
        mask_lrg &= zmag - w1mag > 0.8 * (rmag - zmag) - 0.6  # non-stellar cut
        mask_lrg &= zfibermag < 21.61                   # faint limit
        mask_lrg &= (gmag - w1mag > 2.97) | (rmag - w1mag > 1.8)  # low-z cuts
        mask_lrg &= (
            ((rmag - w1mag > (w1mag - 17.13) * 1.83)
             & (rmag - w1mag > (w1mag - 16.31) * 1.))
            | (rmag - w1mag > 3.4)
        )  # double sliding cuts and high-z extension

    return mask_lrg


def select_elg(cat):
    '''
    columns = ['OBJID', 'BRICKID', 'RELEASE', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FIBERFLUX_G', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z']
    fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/sweep-000p000-010p005.fits'
    cat = Table(fitsio.read(fn, columns=columns))
    mask_elglop, mask_elgvlo = select_elg(cat)
    '''

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    mask_quality = np.full(len(cat), True)

    mask_quality &= (cat['FLUX_IVAR_G'] > 0) & (cat['FLUX_G'] > 0) & (cat['FIBERFLUX_G'] > 0)
    mask_quality &= (cat['FLUX_IVAR_R'] > 0) & (cat['FLUX_R'] > 0)
    mask_quality &= (cat['FLUX_IVAR_Z'] > 0) & (cat['FLUX_Z'] > 0)

    # ADM observed in every band.
    mask_quality &= (cat['NOBS_G'] > 0) & (cat['NOBS_R'] > 0) & (cat['NOBS_Z'] > 0)

    # Apply masks
    maskbits = [1, 12, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))
    mask_quality &= mask_clean

    # gmag = 22.5 - 2.5 * np.log10((cat['FLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))
    # rmag = 22.5 - 2.5 * np.log10((cat['FLUX_R'] / cat['MW_TRANSMISSION_R']).clip(1e-7))
    # zmag = 22.5 - 2.5 * np.log10((cat['FLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))
    # gfibermag = 22.5 - 2.5 * np.log10((cat['FIBERFLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))
    gmag = 22.5 - 2.5 * np.log10(np.clip(cat['FLUX_G']*10**(0.4*3.214*cat['EBV']), 1e-7, None))
    rmag = 22.5 - 2.5 * np.log10(np.clip(cat['FLUX_R']*10**(0.4*2.165*cat['EBV']), 1e-7, None))
    zmag = 22.5 - 2.5 * np.log10(np.clip(cat['FLUX_Z']*10**(0.4*1.211*cat['EBV']), 1e-7, None))
    gfibermag = 22.5 - 2.5 * np.log10(np.clip(cat['FIBERFLUX_G']*10**(0.4*3.214*cat['EBV']), 1e-7, None))

    mask_elglop = mask_quality.copy()

    mask_elglop &= gmag > 20                       # bright cut.
    mask_elglop &= rmag - zmag > 0.15                  # blue cut.
    mask_elglop &= gfibermag < 24.1  # faint cut.
    mask_elglop &= gmag - rmag < 0.5*(rmag - zmag) + 0.1  # remove stars, low-z galaxies.

    mask_elgvlo = mask_elglop.copy()

    # ADM low-priority OII flux cut.
    mask_elgvlo &= gmag - rmag < -1.2*(rmag - zmag) + 1.6
    mask_elgvlo &= gmag - rmag >= -1.2*(rmag - zmag) + 1.3

    # ADM high-priority OII flux cut.
    mask_elglop &= gmag - rmag < -1.2*(rmag - zmag) + 1.3

    return mask_elglop, mask_elgvlo


def select_elg_simplified(cat):

    gmag = cat['gmag']
    rmag = cat['rmag']
    zmag = cat['zmag']
    gfibermag = cat['gfibermag']

    mask_quality = np.isfinite(gmag) & np.isfinite(rmag) & np.isfinite(zmag) & np.isfinite(gfibermag)

    mask_elglop = mask_quality.copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        mask_elglop &= gmag > 20                       # bright cut.
        mask_elglop &= rmag - zmag > 0.15                  # blue cut.
        mask_elglop &= gfibermag < 24.1  # faint cut.
        mask_elglop &= gmag - rmag < 0.5*(rmag - zmag) + 0.1  # remove stars, low-z galaxies.

        mask_elgvlo = mask_elglop.copy()

        # ADM low-priority OII flux cut.
        mask_elgvlo &= gmag - rmag < -1.2*(rmag - zmag) + 1.6
        mask_elgvlo &= gmag - rmag >= -1.2*(rmag - zmag) + 1.3

        # ADM high-priority OII flux cut.
        mask_elglop &= gmag - rmag < -1.2*(rmag - zmag) + 1.3

    return mask_elglop, mask_elgvlo

def select_bgs_bright(cat,field='south'):
    return select_bgs(cat,field)[0]

def select_bgs(cat, field='south'):
    '''
    columns = ['OBJID', 'BRICKID', 'RELEASE', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FIBERFLUX_R', 'FIBERTOTFLUX_R', 'GAIA_PHOT_G_MEAN_MAG', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'REF_CAT']
    fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/sweep-000p000-010p005.fits'
    cat = Table(fitsio.read(fn, columns=columns))
    mask_bgs_bright, mask_bgs_faint = select_bgs(cat)
    '''

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    # Use undereddened Gaia and r-band fluxes
    # grr = cat['GAIA_PHOT_G_MEAN_MAG'] - 22.5 + 2.5*np.log10(1e-16)
    # ii = cat['FLUX_R'] > 0
    # grr[ii] = cat['GAIA_PHOT_G_MEAN_MAG'][ii] - 22.5 + 2.5*np.log10(cat['FLUX_R'][ii])

    # Dereddening the fluxes
    cat['FLUX_G'] *= 10**(0.4*3.214*cat['EBV'])
    cat['FLUX_R'] *= 10**(0.4*2.165*cat['EBV'])
    cat['FLUX_Z'] *= 10**(0.4*1.211*cat['EBV'])
    cat['FLUX_W1'] *= 10**(0.4*0.184*cat['EBV'])
    cat['FIBERFLUX_R'] *= 10**(0.4*2.165*cat['EBV'])

    g = 22.5 - 2.5*np.log10(cat['FLUX_G'].clip(1e-16))
    r = 22.5 - 2.5*np.log10(cat['FLUX_R'].clip(1e-16))
    z = 22.5 - 2.5*np.log10(cat['FLUX_Z'].clip(1e-16))
    w1 = 22.5 - 2.5*np.log10(cat['FLUX_W1'].clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(cat['FIBERFLUX_R'].clip(1e-16))

    mask_quality = np.full(len(cat), True)

    mask_quality &= (cat['NOBS_G'] > 0) & (cat['NOBS_R'] > 0) & (cat['NOBS_Z'] > 0)
    mask_quality &= (cat['FLUX_IVAR_G'] > 0) & (cat['FLUX_IVAR_R'] > 0) & (cat['FLUX_IVAR_Z'] > 0)
    # mask_quality &= ((grr > 0.6) | (cat['GAIA_PHOT_G_MEAN_MAG']==0))

    fmc = np.full(len(cat), False)
    fmc |= ((rfib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((rfib < 22.9) & (r < 20.0) & (r > 17.8))
    fmc |= ((rfib < 2.9 + r) & (r > 20))
    mask_quality &= fmc

    # the SGA galaxies.
    # mask_sga = np.array([(rc[0] == "L") if len(rc) > 0 else False for rc in cat['REF_CAT']])
    # mask_quality |= mask_sga

    # Apply masks
    maskbits = [1, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))
    mask_quality &= mask_clean

    mask_bgs = mask_quality.copy()

    if field=='south':
        mask_bgs &= cat['FLUX_R'] > cat['FLUX_G'] * 10**(-1.0/2.5)
        mask_bgs &= cat['FLUX_R'] < cat['FLUX_G'] * 10**(4.0/2.5)
        mask_bgs &= cat['FLUX_Z'] > cat['FLUX_R'] * 10**(-1.0/2.5)
        mask_bgs &= cat['FLUX_Z'] < cat['FLUX_R'] * 10**(4.0/2.5)
    else:
        mask_bgs &= cat['FLUX_R'] > cat['FLUX_G'] * 10**(-1.0/2.5)
        mask_bgs &= cat['FLUX_R'] < cat['FLUX_G'] * 10**(4.0/2.5)
        mask_bgs &= cat['FLUX_Z'] > cat['FLUX_R'] * 10**(-1.0/2.5)
        mask_bgs &= cat['FLUX_Z'] < cat['FLUX_R'] * 10**(4.0/2.5)

    # BASS r-mag offset with DECaLS.
    offset = 0.04

    mask_bgs_bright = mask_bgs.copy()
    if field=='south':
        mask_bgs_bright &= cat['FLUX_R'] > 10**((22.5-19.5)/2.5)
        mask_bgs_bright &= cat['FLUX_R'] <= 10**((22.5-12.0)/2.5)
        mask_bgs_bright &= cat['FIBERTOTFLUX_R'] <= 10**((22.5-15.0)/2.5)
    else:
        mask_bgs_bright &= cat['FLUX_R'] > 10**((22.5-(19.5+offset))/2.5)
        mask_bgs_bright &= cat['FLUX_R'] <= 10**((22.5-12.0)/2.5)
        mask_bgs_bright &= cat['FIBERTOTFLUX_R'] <= 10**((22.5-15.0)/2.5)

    mask_bgs_faint = mask_bgs.copy()
    if field=='south':
        mask_bgs_faint &= cat['FLUX_R'] > 10**((22.5-20.175)/2.5)
        mask_bgs_faint &= cat['FLUX_R'] <= 10**((22.5-19.5)/2.5)
        schlegel_color = (z - w1) - 3/2.5 * (g - r) + 1.2
        rfibcol = (rfib < 20.75) | ((rfib < 21.5) & (schlegel_color > 0.))
        mask_bgs_faint &= (rfibcol)
    else:
        mask_bgs_faint &= cat['FLUX_R'] > 10**((22.5-(20.220))/2.5)
        mask_bgs_faint &= cat['FLUX_R'] <= 10**((22.5-(19.5+offset))/2.5)
        schlegel_color = (z - w1) - 3/2.5 * (g - (r-offset)) + 1.2
        rfibcol = (rfib < 20.75+offset) | ((rfib < 21.5+offset) & (schlegel_color > 0.))
        mask_bgs_faint &= (rfibcol)

    return mask_bgs_bright, mask_bgs_faint

def select_bgs_bright_individual_cuts(cat, field='south'):
    '''
    columns = ['OBJID', 'BRICKID', 'RELEASE', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FIBERFLUX_R', 'FIBERTOTFLUX_R', 'GAIA_PHOT_G_MEAN_MAG', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'REF_CAT']
    fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/sweep-000p000-010p005.fits'
    cat = Table(fitsio.read(fn, columns=columns))
    mask_bgs_bright, mask_bgs_faint = select_bgs(cat)
    '''

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    # Use undereddened Gaia and r-band fluxes
    # grr = cat['GAIA_PHOT_G_MEAN_MAG'] - 22.5 + 2.5*np.log10(1e-16)
    # ii = cat['FLUX_R'] > 0
    # grr[ii] = cat['GAIA_PHOT_G_MEAN_MAG'][ii] - 22.5 + 2.5*np.log10(cat['FLUX_R'][ii])

    # Dereddening the fluxes
    cat['FLUX_G'] *= 10**(0.4*3.214*cat['EBV'])
    cat['FLUX_R'] *= 10**(0.4*2.165*cat['EBV'])
    cat['FLUX_Z'] *= 10**(0.4*1.211*cat['EBV'])
    cat['FLUX_W1'] *= 10**(0.4*0.184*cat['EBV'])
    cat['FIBERFLUX_R'] *= 10**(0.4*2.165*cat['EBV'])


    g = 22.5 - 2.5*np.log10(cat['FLUX_G'].clip(1e-16))
    r = 22.5 - 2.5*np.log10(cat['FLUX_R'].clip(1e-16))
    z = 22.5 - 2.5*np.log10(cat['FLUX_Z'].clip(1e-16))
    w1 = 22.5 - 2.5*np.log10(cat['FLUX_W1'].clip(1e-16))
    rfib = 22.5 - 2.5*np.log10(cat['FIBERFLUX_R'].clip(1e-16))

    mask_tab = Table()

    fmc = np.full(len(cat), False)
    fmc |= ((rfib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((rfib < 22.9) & (r < 20.0) & (r > 17.8))
    fmc |= ((rfib < 2.9 + r) & (r > 20))
    mask_tab.add_column(fmc,name='FMC')

    # the SGA galaxies.
    # mask_sga = np.array([(rc[0] == "L") if len(rc) > 0 else False for rc in cat['REF_CAT']])
    # mask_quality |= mask_sga

    # Apply masks
    maskbits = [1, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))
    assert(np.all(mask_clean))

    if field=='south':
        mask_tab.add_column(cat['FLUX_R'] > cat['FLUX_G'] * 10**(-1.0/2.5),name='g-r > -1')
        mask_tab.add_column(cat['FLUX_R'] < cat['FLUX_G'] * 10**(4.0/2.5),name='g-r < 4')
        mask_tab.add_column(cat['FLUX_Z'] > cat['FLUX_R'] * 10**(-1.0/2.5),name='r-z > -1')
        mask_tab.add_column(cat['FLUX_Z'] < cat['FLUX_R'] * 10**(4.0/2.5),name='r-z < 4')
    else:
        mask_tab.add_column(cat['FLUX_R'] > cat['FLUX_G'] * 10**(-1.0/2.5),name='g-r > -1')
        mask_tab.add_column(cat['FLUX_R'] < cat['FLUX_G'] * 10**(4.0/2.5),name='g-r < 4')
        mask_tab.add_column(cat['FLUX_Z'] > cat['FLUX_R'] * 10**(-1.0/2.5),name='r-z > -1')
        mask_tab.add_column(cat['FLUX_Z'] < cat['FLUX_R'] * 10**(4.0/2.5),name='r-z < 4')


    # BASS r-mag offset with DECaLS.
    offset = 0.04


    if field=='south':
        mask_tab.add_column(cat['FLUX_R'] > 10**((22.5-19.5)/2.5),name='r < 19.5')
        mask_tab.add_column(cat['FLUX_R'] <= 10**((22.5-12.0)/2.5),name='r > 12')
        mask_tab.add_column(cat['FIBERTOTFLUX_R'] <= 10**((22.5-15.0)/2.5),name='rfibtotmag > 15')

    else:
        mask_tab.add_column(cat['FLUX_R'] > 10**((22.5-(19.5+offset))/2.5),name='r < 19.5')
        mask_tab.add_column(cat['FLUX_R'] <= 10**((22.5-12.0)/2.5),name='r > 12')
        mask_tab.add_column(cat['FIBERTOTFLUX_R'] <= 10**((22.5-15.0)/2.5),name='rfibtotmag > 15')


    return mask_tab

def select_lrg_individual_cuts(cat, field='south'):
    '''
    columns = ['OBJID', 'BRICKID', 'RELEASE', 'RA', 'DEC', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FIBERFLUX_Z', 'FIBERTOTFLUX_Z', 'GAIA_PHOT_G_MEAN_MAG', 'EBV', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z']
    fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/sweep-000p000-010p005.fits'
    cat = Table(fitsio.read(fn, columns=columns))
    mask_lrg = select_lrg(cat)
    '''

    cat = cat.copy()
    cat.rename_columns(cat.colnames, [ii.upper() for ii in cat.colnames])

    mask_tab = Table()

    # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
    mask_tab.add_column(cat['FIBERTOTFLUX_Z'] < 10**(-0.4*(17.5-22.5)),name='zfibertotmag > 17.5')




    # Apply masks
    maskbits = [1, 12, 13]
    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    assert np.all(mask_clean)

    # gmag = 22.5 - 2.5 * np.log10((cat['FLUX_G'] / cat['MW_TRANSMISSION_G']).clip(1e-7))
    # # ADM safe as these fluxes are set to > 0 in notinLRG_mask.
    # rmag = 22.5 - 2.5 * np.log10((cat['FLUX_R'] / cat['MW_TRANSMISSION_R']).clip(1e-7))
    # zmag = 22.5 - 2.5 * np.log10((cat['FLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))
    # w1mag = 22.5 - 2.5 * np.log10((cat['FLUX_W1'] / cat['MW_TRANSMISSION_W1']).clip(1e-7))
    # zfibermag = 22.5 - 2.5 * np.log10((cat['FIBERFLUX_Z'] / cat['MW_TRANSMISSION_Z']).clip(1e-7))

    gmag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_G']*10**(0.4*3.214*cat['EBV']), 1e-7, None))
    rmag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_R']*10**(0.4*2.165*cat['EBV']), 1e-7, None))
    zmag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_Z']*10**(0.4*1.211*cat['EBV']), 1e-7, None))
    w1mag = 22.5 - 2.5*np.log10(np.clip(cat['FLUX_W1']*10**(0.4*0.184*cat['EBV']), 1e-7, None))
    zfibermag = 22.5 - 2.5*np.log10(np.clip(cat['FIBERFLUX_Z']*10**(0.4*1.211*cat['EBV']), 1e-7, None))

    if field=='south':
        mask_tab.add_column(zmag - w1mag > 0.8 * (rmag - zmag) - 0.6,name='non-stellar cut')
        mask_tab.add_column(zfibermag < 21.6,name='faint limit')
        mask_tab.add_column((gmag - w1mag > 2.9) | (rmag - w1mag > 1.8),name='low-z cuts')
        mask_tab.add_column((
            ((rmag - w1mag > (w1mag - 17.14) * 1.8)
             & (rmag - w1mag > (w1mag - 16.33) * 1.))
            | (rmag - w1mag > 3.3)
        ),name="double sliding cuts and high-z extension")
    else:
        mask_tab.add_column(zmag - w1mag > 0.8 * (rmag - zmag) - 0.6,name='non-stellar cut')
        mask_tab.add_column(zfibermag < 21.61,name='faint limit')
        mask_tab.add_column((gmag - w1mag > 2.97) | (rmag - w1mag > 1.8),name='low-z cuts')
        mask_tab.add_column((
            ((rmag - w1mag > (w1mag - 17.13) * 1.83)
             & (rmag - w1mag > (w1mag - 16.31) * 1.))
            | (rmag - w1mag > 3.4)
        ),name="double sliding cuts and high-z extension")

    return mask_tab
