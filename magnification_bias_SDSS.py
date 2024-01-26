#Helper functions for calculating magnification bias in SDSS
#author Lukas Wenzl

#basic imports
from json import load
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
from astropy import units as u

import os


# general data processing for SDSS

b_u, b_g, b_r, b_i, bz = 1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10

b_bands = np.array([b_u, b_g, b_r, b_i, bz])

def load_SDSS_data(filename, catalog_idx=1):
    """Load an sdss catalog.
    example filename: "SDSS_DR12/galaxy_DR12v5_CMASSLOWZTOT_South.fits"
    """
    if isinstance(filename, list):
        print("loading multiple galaxy catalog files. They need to have the same data format")
        lst_data = []
        for file in filename:
            print("loading galaxy catalog {}".format(file))
            hdu = fits.open(file)
            lst_data.append(hdu[catalog_idx].data)
        data = np.concatenate(lst_data)
        return data, None
    else: #only one file
        print("loading galaxy catalog {}".format(filename))
        hdu = fits.open(filename)
        data = hdu[catalog_idx].data
        return data, hdu
    
def apply_redshift_cuts(data, zmin, zmax, redshift_label="Z"):
    mask_lower = data[redshift_label] >= zmin
    mask_upper = data[redshift_label] <= zmax
    mask = mask_lower* mask_upper
    return data[mask]



def get_mag(flux, extinction, clip = True):
    #return -2.5 / np.log(10.) * (np.arcsinh(1e-9 * flux/(2*b_bands)) + np.log(b_bands)) - extinction
    #return -2.5 * np.log10(1e-9 * flux) - extinction
    if(clip):
        return -2.5 * np.log10(1e-9 * flux.clip(min=0.001,max=None)) - extinction
    else:
        print("WARNING: flux clipping turned off")
        return -2.5 * np.log10(1e-9 * flux) - extinction

def get_flux(mag):
    #return 1/1e-9 * 2. * b_bands * np.sinh(-np.log(10.)/2.5 * mag - np.log(b_bands))
    return 1/1e-9 * 10**(-0.4 * mag)

def add_extra_columns_to_data(data):
    """Calculate extra columns for the dataset: extinction corrected magnitudes, extinction corrected fluxes and the extra colour combinations dperp, cperp and cpar.

    Args:
        data: SDSS data set

    Returns:
        data: SDSS data set with extra columns
    """

    data["cModelFlux"] = data['DEVFLUX'] * data['FRACPSF'] + data['EXPFLUX'] * (1. - data['FRACPSF'])
    #EC: extinction corrected
    data["psfMag_EC"] = get_mag(data['PSFFLUX'], data['EXTINCTION'])
    data["fiber2Mag_EC"] = get_mag(data['FIBER2FLUX'], data['EXTINCTION'])
    data["modelMag_EC"]  = get_mag(data['MODELFLUX'], data['EXTINCTION'])
    data["cModelMag_EC"] = get_mag(data["cModelFlux"], data['EXTINCTION'])
    data["dperp"] = (data["modelMag_EC"][:,2] - data["modelMag_EC"][:,3]) - (data["modelMag_EC"][:,1] - data["modelMag_EC"][:,2])/8.
    data["cperp"] = (data["modelMag_EC"][:,2] - data["modelMag_EC"][:,3]) - (data["modelMag_EC"][:,1] - data["modelMag_EC"][:,2])/4.-0.18
    data["cpar"] = 0.7 * (data["modelMag_EC"][:,1] - data["modelMag_EC"][:,2]) + 1.2 * (data["modelMag_EC"][:,2] - data["modelMag_EC"][:,3] - 0.18)
    #extinction corrected cModel fluxes
    data["cModelFlux_EC"] = get_flux(data["cModelMag_EC"] )
    #extinction corrected fiber2 fluxes
    data["fiber2Flux_EC"] = get_flux(data["fiber2Mag_EC"])
    #exctinction corrected model fluxes
    data["modelFlux_EC"] = get_flux(data["modelMag_EC"] )
    #extinction corrected psf fluxes
    data["psfFlux_EC"] = get_flux(data["psfMag_EC"] )
    return data


def get_weights(weights_str, data):
    #need to change these if not using SDSS
    if (weights_str == "None"):
        return None
    if (weights_str == "weights"):
        return data["weights"]
    elif(weights_str == "baseline"):
        return (data["WEIGHT_NOZ"]+ data["WEIGHT_CP"] -1.)* data["WEIGHT_SEEING"]*data["WEIGHT_STAR"]
    elif(weights_str == "NOZ_only"):
        return data["WEIGHT_NOZ"]
    elif(weights_str == "baseline_with_FKP"):
        return (data["WEIGHT_NOZ"]+ data["WEIGHT_CP"] -1.)* data["WEIGHT_SEEING"]*data["WEIGHT_STAR"]* data["WEIGHT_FKP"]
    else:
        print("ERROR: weights_str not recognized")
        #return None



##########################################
#   general function to lens the fluxes (apply_lensing_v3) with helper functions

def DeVaucouleurs_intensity(r_e,r):#in arcsec!
    """Calculate the intensity at a given radius. Arbitrary normalization

    Args:
        r_e : characteristic radius
        r : radius samples for the intensity

    Returns:
        array: Intensity array
    """
    I_e = 1
    return I_e * np.exp(-7.669* ( (r/r_e)**(1./4.) -1.))

def Exponential_intensity(r_e,r):#in arcsec!
    """Alternative light profile to estimate systematic error budget

    Args:
        r_e : characteristic radius
        r : radius samples for the intensity

    Returns:
        array: Intensity array
    """
    #calculate the intensity at a given radius
    #r = 2 #fiber is 2 arcsec
    I_e = 1
    #https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    return I_e * np.exp(-1.678* ( (r/r_e) -1.))

#derivative is with respect to the radius
def Ffiber(theta_e, theta_f = 2., use_exp_profile=False):
    """Calculate fiber flux for an intensity profile

    Args:
        theta_e : Characteristic radius of intensity profile
        theta_f (optional): Aperture of the fiber flux. Defaults to 2..
        use_exp_profile (bool, optional): Switch from DeVaucouleurs profile to exponential profile. Defaults to False.

    Returns:
        Flux
    """
    #integrate De Vaucouleurs profile over the fiber
    step = 0.01
    radius_samples = np.arange(0, theta_f+step, step)
    if(not use_exp_profile):
        #default
        intensity = DeVaucouleurs_intensity(theta_e, radius_samples)
    else:
        intensity = Exponential_intensity(theta_e, radius_samples)
    
    integral = np.trapz(intensity*2*np.pi*radius_samples, x=radius_samples)
    
    return 2*np.pi * integral

def get_cor_for_2fiber_mag(theta_e, use_exp_profile = False):
    """Get the correction for the kappa multiplier for the 2arcsec fiber magnitude

    Args:
        theta_e : characteristic radius of galaxy
        use_exp_profile (bool, optional): Switch from DeVaucouleurs profile to exponential profile. Defaults to False.

    Returns:
        float : kappa multiplier
    """
    #get the correction for the kappa multiplier for the 2 arcsec fiber flux
    fiber_sizes = np.arange(1., 5, 0.1)
    Ffiber_array = [Ffiber(theta_e=theta_e, theta_f = i, use_exp_profile=use_exp_profile ) for i in fiber_sizes]
    dFfiber_dtheta = np.gradient(Ffiber_array, fiber_sizes)
    dlnF_dlntheta = fiber_sizes/Ffiber_array * dFfiber_dtheta
    return np.interp(2. , fiber_sizes, dlnF_dlntheta )


def Fpsf(theta_e, sigma ,use_exp_profile):
    """Calculate the psf flux given a galaxy size and psf size

    Args:
        theta_e : characteristic radius of galaxy
        sigma : psf size
        use_exp_profile (bool): Whether to use the exponential profile or DeVaucouleurs profile

    Returns:
        float: Psf flux
    """
    #integrate De Vaucouleurs profile over the psf. Note: not correctly normalized but the normalization constant cancels
    const = 1. #ignoring a bunch of constants that divide out for our purposes
    step = 0.01
    radius_samples = np.arange(0, 20.+step, step) #20 is sufficiently infinity here
    if(not use_exp_profile):
        intensity = DeVaucouleurs_intensity(theta_e, radius_samples)
    else:
        intensity = Exponential_intensity(theta_e, radius_samples)
    
    integral = np.trapz(const*intensity*radius_samples * np.exp(-radius_samples**2/sigma**2 ), x=radius_samples)
    
    return 2*np.pi * integral

def get_dlnFpsf_dlnsimga(theta_e, sigma, use_exp_profile):
    """Log derivative of the psf flux with respect to sigma the size of the psf. Needed for kappa multiplier correction

    Args:
        theta_e : characteristic radius of galaxy
        sigma : psf size
        use_exp_profile : Whether to use the exponential profile or DeVaucouleurs profile

    Returns:
        Log derivative of the psf flux with respect to sigma 
    """
    #log derivative for the correction of the kappa multiplier for the psf flux
    delta_sigma = 0.01
    
    F0 = Fpsf(theta_e, sigma ,use_exp_profile)
    F1 = Fpsf(theta_e, sigma+delta_sigma , use_exp_profile)
    dFpsf_dsigma = (F1 - F0)/delta_sigma
    return  dFpsf_dsigma * sigma/F0
    

def get_psf_correction_interpolator(use_exp_profile = False):
    """Interpolator for the kappa multiplier correction for the psf magnitude

    Args:
        use_exp_profile (bool, optional): Whether to use the exponential profile or DeVaucouleurs profile. Defaults to False.

    Returns:
        psf_correction_interpolator
    """
    #build interpolator in 2d
    step = 0.02
    theta_e_arr = np.arange(0.001, 10+step, step) #0.02
    sigmas_arr = np.arange(0.001, 2+step, step)
    dlnFpsf_dlnsimga_arr = [[get_dlnFpsf_dlnsimga(theta_e_arr[i], sigmas_arr[j], use_exp_profile=use_exp_profile) for i in range(len(theta_e_arr))] for j in range(len(sigmas_arr))]
    psf_correction_interpolator = RegularGridInterpolator((theta_e_arr, sigmas_arr), np.array(dlnFpsf_dlnsimga_arr).T)

    return psf_correction_interpolator  
#always load the interpolator. Takes around 20sec but saves a lot of time later
print("building interpolator for psf correction")
psf_correction_interpolator = get_psf_correction_interpolator()  
psf_correction_interpolator_exp_profile = get_psf_correction_interpolator(use_exp_profile=True) 


    
def apply_lensing_v3(data_local,  kappa, plots=False, plot_output_filename="test.pdf", verbose=False, use_exp_profile=False):
    """Apply a small amount of lensing kappa to the observed magnitudes of the galaxy data. Combines all the functions to correctly apply the lensing for each type of magnitude in SDSS BOSS.

    Args:
        data_local : galaxy data
        kappa (float): lensing kappa
        plots (bool, optional): Whether to plot out the kappa multipliers. Defaults to False.
        plot_output_filename (str, optional): Filename for the kappa multiplier plot. Defaults to "test.pdf".
        verbose (bool, optional): Print out details. Defaults to False.
        use_exp_profile (bool, optional): Switch to using exponential intensity profile. Defaults to False.

    Returns:
        data_local: galaxy data with extra columns for lensed magnitudes named ..._mag
    """

    if(verbose):
        print("applying lensing as 1+2kappa for all")
        print("also applying fiber mag correction")
        print("also applying psf mag correction")
    
    #lens the fluxes
    data_local["cModelFlux_mag"]  = data_local["cModelFlux"] * (1.+2.*kappa)
    data_local["modelFlux_mag"] = data_local["MODELFLUX"] * (1. +2.*kappa)
    
          
    #fiber correction
    theta_e_galaxies = data_local['R_DEV'] *  0.396 #convert pixel to arcsec
    theta_e_arr = np.arange(0.05, 10, 0.01)
    cor_for_2fiber_mag_arr = [get_cor_for_2fiber_mag(theta_e=i, use_exp_profile=use_exp_profile) for i in theta_e_arr]
    fiber_correction = np.interp(theta_e_galaxies, theta_e_arr, cor_for_2fiber_mag_arr)
    data_local["fiber2Flux_mag"] = data_local["FIBER2FLUX"] * (1. +(2.- fiber_correction)*kappa)

    #psf correction
    conv_factor = (2.*np.sqrt(2.*np.log(2.))) #convert FWHM to sigma of the psf
    sigmas = data_local['PSF_FWHM'] / conv_factor #in arcsec, note they are different in each band
    #deal with failures in PSF FWHM. There are 2 galaxies in the sample, *5 bands -> 10 cases
    sigma_failures = sigmas[:,3] <= 0.001
    if(verbose):
        print("dealing with {} bad values in PSF sigma value (all bands combined)".format(np.sum(sigma_failures)))
    sigmas[sigma_failures] = np.mean(sigmas)
    
    if(verbose):
        print("dealing with {} bad values in theta_e for i band".format(np.sum(theta_e_galaxies[:,3]<= 0.001)))
        print("dealing with {} bad values in theta_e for r band".format(np.sum(theta_e_galaxies[:,2]<= 0.001)))
    theta_e_galaxies[theta_e_galaxies<= 0.001] = np.mean(theta_e_galaxies)
    theta_e_galaxies[theta_e_galaxies>= 10.] = np.mean(theta_e_galaxies)


    psf_correction = sigmas * 0. #create empty array
    for i in range(sigmas.shape[1]):
        points = np.array([theta_e_galaxies[:,i].data,sigmas[:,i].data])
        if(not use_exp_profile):
            psf_correction[:, i]  =psf_correction_interpolator(points.T)
        else:
            psf_correction[:, i]  =psf_correction_interpolator_exp_profile(points.T)
    data_local["psfFlux_mag"] = data_local["PSFFLUX"] * (1. +(2.-psf_correction)*kappa)
    #plt.hist(psf_correction[:, 3]) #roughly peaking at 1

    #go to magnitudes
    data_local["fiber2Mag_EC_mag"] = get_mag(data_local['fiber2Flux_mag'], data_local['EXTINCTION'])
    data_local["cModelMag_EC_mag"] = get_mag(data_local['cModelFlux_mag'], data_local['EXTINCTION'])
    data_local["modelMag_EC_mag"] = get_mag(data_local['modelFlux_mag'], data_local['EXTINCTION'])
    data_local["psfMag_EC_mag"] = get_mag(data_local['psfFlux_mag'], data_local['EXTINCTION'])
    
    if(verbose):
        shift_iband = np.array(data_local["cModelMag_EC_mag"][:,3] - data_local["cModelMag_EC"][:,3])
        print("shift in i band for cModelMag: {}".format(np.mean(shift_iband)))
    
    if(plots):
        plt.figure()
        plt.title("fiber correction")
        plt.hist(fiber_correction[:,3], bins=50, density=True)
        #plt.hist(fiber_correction[:,4], bins=50)
        mean = np.mean(fiber_correction[:,3])
        print(mean)
        variance = np.var(fiber_correction[:,3])
        sigma = np.sqrt(variance)
        print(sigma)
        plt.figure()
        plt.title("psf correction")
        plt.hist(psf_correction[:,3], bins=50)
        #plt.hist(psf_correction[:,4], bins=50)
        mean = np.mean(psf_correction[:,3])
        print(mean)
        variance = np.var(psf_correction[:,3])
        sigma = np.sqrt(variance)
        print(sigma)
        
        plt.figure()
        plt.xlabel("$\kappa$ multiplier")
        plt.ylabel("Frequency")
        plt.axvline(2., label="cModelMag, modelMag")
        plt.hist(2.-fiber_correction[:,3], bins=50, density=True, histtype='step',linestyle="--",label="fib2Mag")
        plt.hist(2.-psf_correction[:,3], bins=50, density=True, histtype='step',label="psfMag")
        
        plt.legend()
        plt.savefig(plot_output_filename)
          
    return data_local


#alternative analysis choice of using exponential profiles instead
#def effective_slope_Exp(r):
#    '''
#    Here the radius r is in units of the Exponential radius Rexp
#    '''
#    return r**2 * np.exp(-r) / (1 - np.exp(-r)*(1+r))

#
##########################################################################################
#   all the photometric selections
def reapply_photocuts_CMASS(cModelMag_new, modelMag_new, fiber2Mag_new, psfMag_new, dperp):
    """Function to apply the photometric selection of CMASS. Needs the corresponding columns from the galaxy data

    Returns:
        bool array: selection
        array of bool arrays : selection for each condition
    """
    
    cond1 = dperp > 0.55
    cond2 = cModelMag_new[:, 3] < 19.86 + 1.6 *(dperp -0.8)
    cond3 = (cModelMag_new[:, 3] <19.9) * (cModelMag_new[:, 3] >17.5)
    cond4 = modelMag_new[:,4] - modelMag_new[:, 3] < 2.
    cond5 = fiber2Mag_new[:,3] < 21.5
    cond6 = psfMag_new[:,3] - modelMag_new[:,3] > 0.2 + 0.2*(20.-modelMag_new[:,3])
    cond7 = psfMag_new[:,4] - modelMag_new[:,4] > 9.125 - (0.46*modelMag_new[:,4])
    
    combined = cond1*cond2*cond3*cond4*cond5*cond6*cond7
    each_condition = [cond1,cond2, cond3, cond4, cond5, cond6, cond7]
    
    return combined, each_condition

def reapply_photocuts_LOWZ(cModelMag_new, modelMag_new, fiber2Mag_new, psfMag_new, cperp, cpar):
    """Function to apply the photometric selection of LOWZ. Needs the corresponding columns from the galaxy data

    Returns:
        bool array: selection
        array of bool arrays : selection for each condition
    """
    
    cond1 = np.abs(cperp) < 0.2
    cond2 = cModelMag_new[:, 2] < 13.5 + (cpar/0.3)
    cond3 = (cModelMag_new[:, 2] <19.6) * (cModelMag_new[:, 2] >16.)
    cond4 = psfMag_new[:,2] - cModelMag_new[:,2] > 0.3
    
    combined = cond1*cond2*cond3*cond4
    each_condition = [cond1,cond2, cond3, cond4]
    
    return combined, each_condition

#ugriz
#01234
def reapply_photocuts_z1z3(cModelMag_new, modelMag_new, fiber2Mag_new, psfMag_new, cperp, cpar, dperp, chunk):
    """Function to apply the photometric selection of z1 and z3. Needs the corresponding columns from the galaxy data

    Returns:
        bool array: selection
    """
    combined_LOWZ, _ = reapply_photocuts_LOWZ(cModelMag_new, modelMag_new, fiber2Mag_new, psfMag_new, cperp, cpar)
    combined_CMASS, _ = reapply_photocuts_CMASS(cModelMag_new, modelMag_new, fiber2Mag_new, psfMag_new, dperp)

    #extra conditions for LOWZ only applied to certain chunks
    mask_chunk2 = (chunk == 2)
    mask_chunk3to6 = (chunk > 2) * (chunk < 7)

    chunk2_extra1 = cModelMag_new[:, 2] < 13.4 + (cpar/0.3)
    chunk2_extra2 = cModelMag_new[:, 2] < 19.6
    chunk2_extra3 = psfMag_new[:,3] - modelMag_new[:,3] > 0.2 + 0.2*(20.-modelMag_new[:,3]) 

    extra_chunk2 = np.ones_like(combined_LOWZ)
    tmp = chunk2_extra1*chunk2_extra2*chunk2_extra3
    extra_chunk2[mask_chunk2] = tmp[mask_chunk2] #only applied to chunk2

    chunk3to6_extra1 = cModelMag_new[:, 2] < 13.4 + (cpar/0.3)
    chunk3to6_extra2 = (cModelMag_new[:, 2] < 19.5) * (cModelMag_new[:, 2] >17.)
    chunk3to6_extra3 = psfMag_new[:,3] - modelMag_new[:,3] > 0.2 + 0.2*(20.-modelMag_new[:,3])
    chunk3to6_extra4 = psfMag_new[:,4] - modelMag_new[:,4] > 9.125 - 0.46*modelMag_new[:,4]

    extra_chunk3to6 = np.ones_like(combined_LOWZ)
    tmp = chunk3to6_extra1*chunk3to6_extra2*chunk3to6_extra3*chunk3to6_extra4
    extra_chunk3to6[mask_chunk3to6] = tmp[mask_chunk3to6]

    total = (combined_LOWZ * extra_chunk2 * extra_chunk3to6 + combined_CMASS ) >0.
    return total#total, [combined1, combined2]

#
##########################################
#   R: photoz failures
#   Note: this is not discussed in detail in the paper as we ultimately determined the photoz failure rate is correctly accounted for via the applied weights.

#failure rate as sigmoid. Getting derivative analytically

func_log_success_rate = lambda x, b, c, d : -c/(1.+np.exp(d*(-x +21.5))) +b  #sigmoid
func_log_success_rate_der = lambda x, b, c, d : -c*d* np.exp(d*(-x +21.5))/(1.+np.exp(d*(-x +21.5)))**2  #analytic derivative

fit_success_rate_CMASS = np.array([-0.0084071 ,  0.18640093,  8.23253506])
fit_success_rate_LOWZ = np.array([-3.40221299e-03,  4.42265490e-01,  5.22097162e+00])

def get_R(data_local, use_exp_profile=False, case = "CMASS", weights=None):
    #impact on alpha is R/2N0

    #fiber correction roughly factor 1.6 instead of 2 but depends on R_DEV
    theta_e_galaxies = data_local['R_DEV'][:, 3] *  0.396 #convert pixel to arcsec
    theta_e_arr = np.arange(0.05, 10, 0.01)
    cor_for_2fiber_mag_arr = [get_cor_for_2fiber_mag(theta_e=i, use_exp_profile=use_exp_profile) for i in theta_e_arr]
    fiber_correction = np.interp(theta_e_galaxies, theta_e_arr, cor_for_2fiber_mag_arr)
    #print(np.mean(2.-fiber_correction))
    delta_mag_fib = -2.5 * (2.-fiber_correction) *np.log(10.)**(-1.)
    #derivative of shift in mag with respect to kappa

    fiber2Mag_EC = data_local["fiber2Mag_EC"][:,3] #i band

    if(weights == None):
        weights = np.ones_like(fiber2Mag_EC)
        #multiplying by weights works since R gets divided by N which also gets scaled by the weights
        #this should account for most of the redshift dependence of failures as the reason why higher redshift failure objects fail more is because they are fainter.

    R = None
    if(case == "CMASS"):
        R = np.sum(func_log_success_rate_der(fiber2Mag_EC, *fit_success_rate_CMASS) * delta_mag_fib * weights)
    elif(case == "LOWZ"):
        R = np.sum(func_log_success_rate_der(fiber2Mag_EC, *fit_success_rate_LOWZ) * delta_mag_fib * weights)
    #case z1 and z3 best approach unclear. could use a mask on which are part of which sample but not very accurate since I don't need the redshift dependence of the failure rate. Can only do an estimate. maybe using CMASS > 0.5 and CMASS redshift < 0.5 
    return R  


#
##########################################
#   calculating alpha


def get_alpha(Boolean_change_left, Boolean_change_right, kappa, weights=None):
    """Function to calculate alpha given how many objects fall out of the selection when applying an amount of lensing kappa and -kappa

    Args:
        Boolean_change_left (_type_): Bool array of objects falling out of selection when applying lensing kappa 
        Boolean_change_right (_type_): Bool array of objects falling out of selection when applying lensing -kappa 
        kappa (float): amount of lensing kappa applied
        weights (float array, optional): Weights for each galaxy. Defaults to None.

    Returns:
        float: alpha simple estimate
        float: Poisson uncertainty for alpha

    """
    
    if(weights is None):
        N = len(Boolean_change_left)
        N_change_left = np.sum(Boolean_change_left) - N
        N_change_right = np.sum(Boolean_change_right) - N
    else:
        #accounting for weights
        N = np.sum(weights)
        N_change_left = np.sum(weights[Boolean_change_left]) - N
        N_change_right = np.sum(weights[Boolean_change_right]) - N
    
    #note the minus sign
    alpha = (N_change_left - N_change_right )/N * 1. /(2.*kappa)
    #shot noise (incorrect)
    alpha_error = np.sqrt(np.abs(N_change_left)+np.abs(N_change_right))/N * 1./(2.*kappa) #shot noise error
    
    #this neglects contribution from N0 ! only very minor difference
    error_from_N0 = alpha/np.sqrt(N)
    alpha_error_full = np.sqrt(alpha_error**2 + error_from_N0**2)
    print("base error: {}".format(alpha_error))
    print("N0 error: {}".format(error_from_N0))
    print("combined error: {}".format(alpha_error_full))

    return alpha, alpha_error

def get_dN(lst_change_left, lst_change_right, weights=None):
    """Helper function to get the changes in the sample for each step in the binwise estimate
    """

    n_bins = len(lst_change_left)
    dNs = np.zeros(n_bins)
    dNs_error = np.zeros(n_bins)
    dNs_bins = np.zeros(n_bins-1)
    dNs_bins_error = np.zeros(n_bins-1)

    previous_N_change_left = 0
    previous_N_change_right = 0
    previous_N_error_left = 0
    previous_N_error_right = 0
    for i in range(n_bins):
        Boolean_change_left = lst_change_left[i]
        Boolean_change_right = lst_change_right[i]
        if(weights is None):
            N = len(Boolean_change_left)
            N_change_left = np.sum(Boolean_change_left) - N
            N_change_right = np.sum(Boolean_change_right) - N
            dNs[i] = N_change_left - N_change_right
            dNs_error[i] = np.sqrt(np.abs(N_change_left)+np.abs(N_change_right))
            if(i>0):
                dNs_bins[i-1] = dNs[i] - dNs[i-1]
                dNs_bins_error[i-1] = np.sqrt(np.abs(N_change_left-previous_N_change_left) + np.abs(N_change_right-previous_N_change_right))
        else:
            #accounting for weights
            N = np.sum(weights)
            N_change_left = np.sum(weights[Boolean_change_left]) - N
            N_change_right = np.sum(weights[Boolean_change_right]) - N
            N_error_left = np.sum(weights[Boolean_change_left]**2) - np.sum(weights**2) #only counting the objects that fall out under lensing
            N_error_right = np.sum(weights[Boolean_change_right]**2) - np.sum(weights**2)
            #todo
            dNs[i] = N_change_left - N_change_right
            dNs_error[i] = np.sqrt(np.abs(N_error_left) + np.abs(N_error_right)) #poisson error with weights, for weights = 1 returning to standard Poisson error
            if(i>0):
                dNs_bins[i-1] = dNs[i] - dNs[i-1]
                #now need to use the weights in the bin
                dNs_bins_error[i-1] = np.sqrt(np.abs(N_error_left-previous_N_error_left) + np.abs(N_error_right-previous_N_error_right))
            #saving the previous values to allow for the binwise estimates
            previous_N_error_left = N_error_left
            previous_N_error_right = N_error_right
        previous_N_change_left = N_change_left
        previous_N_change_right = N_change_right

    return dNs, dNs_error, dNs_bins, dNs_bins_error

#calculate alpha from a single step size
def calculate_alpha_simple_CMASS(data, kappa, lensing_func =apply_lensing_v3 , show_each_condition=True, weights_str="baseline", use_exp_profile=False): 
    """Function to calculate the simple estimate for alpha for CMASS

    Args:
        data : galaxy data
        kappa (float): kappa step size
        lensing_func (func, optional): Function to apply lensing to the data. Defaults to apply_lensing_v3.
        show_each_condition (bool, optional): Print out more details. Defaults to True.
        weights_str (str, optional): Weights for each galaxy. Defaults to "baseline".
        use_exp_profile (bool, optional): Switch to using the exponential profile. Defaults to False.

    Returns:
        float: simple alpha estimate
        float: poisson error for alpha estimate
    """
    
    #assuming kappa positive
    weights = get_weights(weights_str, data)
    #postivite kappa: increase #gal at faint end. 
    #convention: left-sided derivative on the faint end. So need minus sign
    data = lensing_func(data,  kappa, use_exp_profile=use_exp_profile)
    combined_left, each_condition_left = reapply_photocuts_CMASS(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], dperp= data["dperp"])
    
    #other side
    data = lensing_func(data,  -1.*kappa, use_exp_profile=use_exp_profile)
    combined_right, each_condition_right = reapply_photocuts_CMASS(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], dperp= data["dperp"])
    
    alpha, alpha_error = get_alpha(combined_left, combined_right, kappa, weights=weights)
    print("-------")
    print("Overall alpha without R= {}".format(alpha))

    R = get_R(data, use_exp_profile=use_exp_profile, case = "CMASS")
    print("R = {} (not added)".format(R))

    
    if(show_each_condition):
        checksum = 0
        for i in range(len(each_condition_left)):
            #with weights the alpha_i are right but not the number removed, they don't include the weights
            alpha_i, alpha_i_error = get_alpha(each_condition_left[i], each_condition_right[i], kappa, weights=weights)
            checksum += alpha_i
            N = len(each_condition_left[i])
            print("Condition {} shifts alpha by {}. left removed {}, right removed {} gal".format(i+1, alpha_i, np.sum(each_condition_left[i])-N, np.sum(each_condition_right[i])-N))
        print("checksum = {}".format(checksum))
    return alpha, alpha_error

#calculate alpha from multiple step sizes
def calculate_alpha_CMASS(data, kappas, lensing_func =apply_lensing_v3 , weights_str="baseline", use_exp_profile=False): 
    """Baseline magnification bias estimate with our binwise estimator for CMASS.

    Args:
        data (_type_): galaxy data
        kappas (array): array of kappa steps
        lensing_func (func, optional): Function to apply lensing kappa to the data. Defaults to apply_lensing_v3.
        weights_str (array, optional): Weights for each galaxy. Using a string to select cases. Defaults to "baseline".
        use_exp_profile (bool, optional): Switch to using an exponential profile. Defaults to False.

    Returns:
        dictionary: result for the magnification bias estimate
    """
    #assuming kappa positive
    weights = get_weights(weights_str, data)
    #change in N for each kappa bin. 
    dNs = np.zeros_like(kappas)
    if(weights is None):
        N0 = len(data["cModelMag_EC"])
    else:
        N0 = np.sum(weights)
    kappa_step = kappas[1] - kappas[0] #assuming uniform spacing
    print("starting CMASS")

    if(weights is not None):
        print("weighted mean redshift = {:.3f}".format(np.average(data["Z"], weights=weights)))

    assert kappas[0] == 0. , print("Error: kappa list should start with zero, kappas[0] = {}".format(kappas[0]))

    #need to save the full information of which galaxies are removed especially for the case with weights
    lst_left = []
    lst_right = []
    for i, kappa in enumerate(kappas):
        #print(kappa)
        #postivite kappa: increase #gal at faint end. 
        #convention: left-sided derivative on the faint end. So need a minus sign
        data = lensing_func(data,  kappa, use_exp_profile=use_exp_profile)
        combined_left, each_condition_left = reapply_photocuts_CMASS(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], dperp= data["dperp"])
        
        #other side
        data = lensing_func(data,  -1.*kappa, use_exp_profile=use_exp_profile)
        combined_right, each_condition_right = reapply_photocuts_CMASS(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], dperp= data["dperp"])
        
        lst_left.append(combined_left)
        lst_right.append(combined_right)

    dNs, dNs_error, dNs_bins, dNs_bins_error = get_dN(lst_left, lst_right, weights=weights)

    R = get_R(data, use_exp_profile=False, case = "LOWZ")
    print("R = {}".format(R))

    R_over2N0 = R/(2.*N0)
    print("R/2N= {}".format(R_over2N0))
        
    result = {}
    result["dNs"] = dNs
    result["kappas"] = kappas
    result["N0"] = N0

    result["R"] = R
    result["R_over2N0"] = R_over2N0

    #dNs_bins = dNs[1:] - dNs[:-1]
    result["dNs_bins"] = dNs_bins

    norm = 1/(N0 * 2 * kappa_step)
    result["norm"] = norm
    result["As"]  = dNs_bins * norm

    #result["As_error_neglectingN0term"] = 
    result["As_error"] = dNs_bins_error*norm
    #result["As_error_withcorrN0term"] =  np.sqrt(np.abs(dNs_bins)*(norm)**2 + (np.abs(dNs_bins)*norm)**2/N0) #not working anymore with weights but checked that the N0 term is in fact negligible at the percent level

    #getting the simple estimates for free
    print("Not including R in the alpha simple estimate")
    result["alpha_simple"] = (dNs[1:] / (N0 * 2* kappas[1:])) #+R_over2N0
    result["alpha_simple_error"] = dNs_error[1:] / (N0 * 2* kappas[1:])
    result["alpha_simple_error_full"] = np.sqrt((dNs_error[1:]**2 / (N0 * 2* kappas[1:])**2) + (result["alpha_simple"]**2/N0))

    # ii_min = 0
    # ii_max = 20#20#8
    #xdats = result["kappas"][1:]#[ii_min:ii_max]
    xdats = result["kappas"][1:]-(kappa_step/2.)#[ii_min:ii_max]
    ydats = result["As"]#[ii_min:ii_max]
    sigmas = result["As_error"]#[ii_min:ii_max]
    result["fit"] = fit_linear(xdats, ydats, sigmas)

    
    return result

def calculate_alpha_simple_LOWZ(data, kappa, lensing_func =apply_lensing_v3 , show_each_condition=True, weights_str='baseline'): 
    """Function to calculate the simple estimate for alpha for LOWZ

    Args:
        data : galaxy data
        kappa (float): kappa step size
        lensing_func (func, optional): Function to apply lensing to the data. Defaults to apply_lensing_v3.
        show_each_condition (bool, optional): Print out more details. Defaults to True.
        weights_str (str, optional): Weights for each galaxy. Defaults to "baseline".
        use_exp_profile (bool, optional): Switch to using the exponential profile. Defaults to False.

    Returns:
        float: simple alpha estimate
        float: poisson error for alpha estimate
    """

    #assuming kappa positive
    #postivite kappa: increase #gal at faint end. 
    #convention: left-sided derivative on the faint end. So need minus sign
    weights = get_weights(weights_str, data)
    data = lensing_func(data,  kappa)
    combined_left, each_condition_left = reapply_photocuts_LOWZ(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"])
    
    #other side
    data = lensing_func(data,  -1.*kappa)
    combined_right, each_condition_right = reapply_photocuts_LOWZ(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"])
    

    alpha, alpha_error = get_alpha(combined_left, combined_right, kappa, weights=weights)
    print("-------")
    print("Overall alpha without R= {}".format(alpha))

    R = get_R(data, use_exp_profile=False, case = "LOWZ")
    print("R = {} (not added)".format(R))

    if(show_each_condition):
        checksum = 0
        for i in range(len(each_condition_left)):
            alpha_i, alpha_i_error = get_alpha(each_condition_left[i], each_condition_right[i], kappa, weights=weights)
            checksum += alpha_i
            N = len(each_condition_left[i])
            print("Condition {} shifts alpha by {}. left removed {}, right removed {} gal".format(i+1, alpha_i, np.sum(each_condition_left[i])-N, np.sum(each_condition_right[i])-N))
        print("checksum = {}".format(checksum))
    return alpha, alpha_error

#calculate alpha from multiple step sizes
def calculate_alpha_LOWZ(data, kappas, lensing_func =apply_lensing_v3 , weights_str="baseline", use_exp_profile=False): 
    """Baseline magnification bias estimate with our binwise estimator for LOWZ.

    Args:
        data (_type_): galaxy data
        kappas (array): array of kappa steps
        lensing_func (func, optional): Function to apply lensing kappa to the data. Defaults to apply_lensing_v3.
        weights_str (array, optional): Weights for each galaxy. Using a string to select cases. Defaults to "baseline".
        use_exp_profile (bool, optional): Switch to using an exponential profile. Defaults to False.

    Returns:
        dictionary: result for the magnification bias estimate
    """
    #assuming kappa positive
    #change in N for each kappa bin. 
    weights = get_weights(weights_str, data)
    dNs = np.zeros_like(kappas)
    if(weights is None):
        N0 = len(data["cModelMag_EC"])
    else:
        N0 = np.sum(weights)
    kappa_step = kappas[1] - kappas[0] #assuming uniform spacing

    #need to save the full information for which galaxies are removed especially for the case with weights
    lst_left = []
    lst_right = []
    print("starting LOWZ")
    if(weights is not None):
        print("weighted mean redshift = {:.3f}".format(np.average(data["Z"], weights=weights)))

    for i, kappa in enumerate(kappas):
        #print(kappa)
        #postivite kappa: increase #gal at faint end. 
        #convention: left-sided derivative on the faint end. So need minus sign
        data = lensing_func(data,  kappa, use_exp_profile=use_exp_profile)
        combined_left, each_condition_left = reapply_photocuts_LOWZ(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"])
        
        #other side
        data = lensing_func(data,  -1.*kappa, use_exp_profile=use_exp_profile)
        combined_right, each_condition_right = reapply_photocuts_LOWZ(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"])
        
        lst_left.append(combined_left)
        lst_right.append(combined_right)

    dNs, dNs_error, dNs_bins, dNs_bins_error = get_dN(lst_left, lst_right, weights=weights)

    R = get_R(data, use_exp_profile=False, case = "LOWZ")
    print("R = {}".format(R))

    R_over2N0 = R/(2.*N0)
    print("R/2N= {}".format(R_over2N0))
        
    result = {}
    result["dNs"] = dNs
    result["kappas"] = kappas
    result["N0"] = N0

    result["R"] = R
    result["R_over2N0"] = R_over2N0

    #dNs_bins = dNs[1:] - dNs[:-1]
    result["dNs_bins"] = dNs_bins

    norm = 1/(N0 * 2 * kappa_step)
    result["norm"] = norm
    result["As"]  = dNs_bins * norm

    #result["As_error_neglectingN0term"] = 
    result["As_error"] = dNs_bins_error*norm
    #result["As_error_withcorrN0term"] =  np.sqrt(np.abs(dNs_bins)*(norm)**2 + (np.abs(dNs_bins)*norm)**2/N0) 

    #getting the simple estimates for free
    result["alpha_simple"] = dNs[1:] / (N0 * 2* kappas[1:])
    result["alpha_simple_error"] = dNs_error[1:] / (N0 * 2* kappas[1:])
    result["alpha_simple_error_full"] = np.sqrt((dNs_error[1:]**2 / (N0 * 2* kappas[1:])**2) + (result["alpha_simple"]**2/N0))


    # ii_min = 0
    # ii_max = 20#20#8
    #xdats = result["kappas"][1:]#[ii_min:ii_max]
    xdats = result["kappas"][1:]-(kappa_step/2.)#[ii_min:ii_max]
    ydats = result["As"]#[ii_min:ii_max]
    sigmas = result["As_error"]#[ii_min:ii_max]
    result["fit"] = fit_linear(xdats, ydats, sigmas)
    
    return result
    

def calculate_alpha_simple_z1z3(data, kappa, lensing_func =apply_lensing_v3 , show_each_condition=True, weights_str="baseline"): 
    """Function to calculate the simple estimate for alpha for z1 or z3

    Args:
        data : galaxy data
        kappa (float): kappa step size
        lensing_func (func, optional): Function to apply lensing to the data. Defaults to apply_lensing_v3.
        show_each_condition (bool, optional): Print out more details. Defaults to True.
        weights_str (str, optional): Weights for each galaxy. Defaults to "baseline".
        use_exp_profile (bool, optional): Switch to using the exponential profile. Defaults to False.

    Returns:
        float: simple alpha estimate
        float: poisson error for alpha estimate
    """

    #assuming kappa positive
    #postivite kappa: increase #gal at faint end. 
    #convention: left-sided derivative on the faint end. So need minus sign
    weights = get_weights(weights_str, data)
    data = lensing_func(data,  kappa)
    combined_left = reapply_photocuts_z1z3(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"], dperp=data["dperp"], chunk=data["ICHUNK"])
    
    #other side
    data = lensing_func(data,  -1.*kappa)
    combined_right = reapply_photocuts_z1z3(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"], dperp=data["dperp"], chunk=data["ICHUNK"])
    

    alpha, alpha_error = get_alpha(combined_left, combined_right, kappa, weights=weights)
    print("-------")
    print("Overall alpha= {}".format(alpha))

    return alpha, alpha_error

#calculate alpha from multiple step sizes
def calculate_alpha_z1z3(data, kappas, lensing_func =apply_lensing_v3 , weights_str="baseline", use_exp_profile=False): 
    """Baseline magnification bias estimate with our binwise estimator for z1 and z3.

    Args:
        data (_type_): galaxy data
        kappas (array): array of kappa steps
        lensing_func (func, optional): Function to apply lensing kappa to the data. Defaults to apply_lensing_v3.
        weights_str (array, optional): Weights for each galaxy. Using a string to select cases. Defaults to "baseline".
        use_exp_profile (bool, optional): Switch to using an exponential profile. Defaults to False.

    Returns:
        dictionary: result for the magnification bias estimate
    """
    #assuming kappa positive
    #change in N for each kappa bin. 
    weights = get_weights(weights_str, data)
    dNs = np.zeros_like(kappas)
    if(weights is None):
        N0 = len(data["cModelMag_EC"])
    else:
        N0 = np.sum(weights)
    kappa_step = kappas[1] - kappas[0] #assuming uniform spacing

    #need to save the full information for which galaxies are removed especially for the case with weights
    lst_left = []
    lst_right = []

    print("starting z1z3")
    if(weights is not None):
        print("weighted mean redshift = {:.3f}".format(np.average(data["Z"], weights=weights)))
    for i, kappa in enumerate(kappas):
        #print(kappa)
        #postivite kappa: increase #gal at faint end. 
        #convention: left-sided derivative on the faint end. So need minus sign
        data = lensing_func(data,  kappa, use_exp_profile=use_exp_profile)
        combined_left = reapply_photocuts_z1z3(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"], dperp=data["dperp"], chunk=data["ICHUNK"])
        
        #other side
        data = lensing_func(data,  -1.*kappa, use_exp_profile=use_exp_profile)
        combined_right = reapply_photocuts_z1z3(cModelMag_new= data["cModelMag_EC_mag"], modelMag_new = data["modelMag_EC_mag"], fiber2Mag_new= data["fiber2Mag_EC_mag"], psfMag_new= data["psfMag_EC_mag"], cperp= data["cperp"], cpar=data["cpar"], dperp=data["dperp"], chunk=data["ICHUNK"])
        
        lst_left.append(combined_left)
        lst_right.append(combined_right)

    dNs, dNs_error, dNs_bins, dNs_bins_error = get_dN(lst_left, lst_right, weights=weights)
        
    #print("Reminder: not modeling R!")
    result = {}
    result["dNs"] = dNs
    result["kappas"] = kappas
    result["N0"] = N0

    #dNs_bins = dNs[1:] - dNs[:-1]
    result["dNs_bins"] = dNs_bins

    norm = 1/(N0 * 2 * kappa_step)
    result["norm"] = norm
    result["As"]  = dNs_bins * norm

    #result["As_error_neglectingN0term"] = 
    result["As_error"] = dNs_bins_error*norm
    #result["As_error_withcorrN0term"] =  np.sqrt(np.abs(dNs_bins)*(norm)**2 + (np.abs(dNs_bins)*norm)**2/N0) 

    #getting the simple estimates for free
    result["alpha_simple"] = dNs[1:] / (N0 * 2* kappas[1:])
    result["alpha_simple_error"] = dNs_error[1:] / (N0 * 2* kappas[1:])
    #result["alpha_simple_error_full"] = np.sqrt((dNs_error[1:]**2 / (N0 * 2* kappas[1:])**2) + ((dNs_error[1:]**2 / (N0**2 * 2* kappas[1:])  )**2 * N0))
    result["alpha_simple_error_full"] = np.sqrt((dNs_error[1:]**2 / (N0 * 2* kappas[1:])**2) + (result["alpha_simple"]**2/N0))

    # ii_min = 0
    # ii_max = 20#20#8
    #xdats = result["kappas"][1:]#[ii_min:ii_max]
    xdats = result["kappas"][1:]-(kappa_step/2.)#[ii_min:ii_max]
    ydats = result["As"]#[ii_min:ii_max]
    sigmas = result["As_error"]#[ii_min:ii_max]
    result["fit"] = fit_linear(xdats, ydats, sigmas)
    
    return result


def get_uncertainty_band(func, xdats, res, cov):
    """Get uncertainty band for plots
    """
    n_samples = 1000
    samples = np.random.multivariate_normal(res, cov, size=n_samples)
    
    #error band
    data_arr = np.zeros((n_samples, len(xdats)))
    for i in range(n_samples):
        data_arr[i] = func(xdats, *samples[i])
    data_mean = np.mean(data_arr, axis=0)
    data_std = np.std(data_arr,axis=0)
    return data_mean, data_std

def fit_linear(xdats, ydats, sigmas):
    """Fit a line to the data

    Returns:
        dictionary: result of the fit
    """
    fit = {}
    #fitting a line through all points
    func = lambda x, a, c: a*x +  c
    res, cov = curve_fit(func, xdats, ydats, sigma=sigmas, absolute_sigma=True)# this is very subtle!!!
    fit["xdats"] = xdats
    fit["ydats"] = ydats
    fit["sigmas"] = sigmas
    fit["alpha_fit"] = res[1]
    fit["alpha_fit_error"] = np.sqrt(cov[1,1])
    fit["slope_fit"] = res[0]
    fit["slope_fit_error"] = np.sqrt(cov[0,0])
    fit["cov"] = cov

    chi2 =  np.sum((ydats - func(xdats,*res))**2 / sigmas**2)
    n_bins = len(ydats)
    dof = n_bins - len(res)#(ii_max - ii_min - len(res))
    red_chi2 =  chi2 / dof
    print(chi2, red_chi2)
    fit["chi2"] = chi2
    fit["red_chi2"] = red_chi2
    from scipy import stats
    PTE = stats.chi2.sf(chi2, dof)
    fit["dof"] = dof
    fit["PTE"] = PTE
    print("PTE = {}".format(PTE))
    from scipy.special import erfinv
    fit["PTE_in_sigma"] = erfinv(1. -PTE ) * np.sqrt(2.)
    return fit