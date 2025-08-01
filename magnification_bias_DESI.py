# To apply the magnification bias estimate to another survey you need:
# * The galaxy catalog including the magnitudes used for the photometric selection
# * The exact conditions used for the photometric selection
# * An understanding of how the magnitudes used behave under lensing. In our work for SDSS BOSS we characterized this for magnitudes that capture the full light of the galaxy, psf magnitudes and aperture magnitudes. If you need other magnitudes you need to characterize them yourself.

#Below we lay out empty functions you need to fill in for your specific survey. You essentially need to fill out one function for each of the bullet points above.

#For an example see magnification_bias_SDSS.py

import numpy as np
from istarget import get_required_columns
from astropy.io import fits
from astropy.table import Table,join,hstack,vstack
import os
import fitsio
from scipy.optimize import curve_fit
from cuts import apply_photocuts_DESI,apply_magnitude_cuts,apply_secondary_cuts,apply_photocuts_DESI_individual_cuts,apply_secondary_cuts_individual_cuts
import json
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import NearestNDInterpolator



import copy

from load_DESI_catalogues import read_table

def load_survey_data(galaxy_type,config,zmin=None,zmax=None,debug=False):
    fpath_lss = config['general']['full_lss_path']
    # fpath_gal = config['general']['lensing_path']
    fpath_gal = fpath_lss
    version = config['general']['version']

    required_columns = get_required_columns(galaxy_type)

    # load our catalogue that contains the clean sample
    load_columns = ["TARGETID","Z"] + required_columns
    if (galaxy_type == "BGS_BRIGHT") and ('Y1' in fpath_lss):
        load_columns += ["ABSMAG01_SDSS_R"]
    if config['general']['fiber_mag_lensing'] == 'Tabulated':
        tabulatedbool=True
    else:
        tabulatedbool=False
    if "DA2" in config['general']['full_lss_path']:
        gal_tab = read_table(fpath_gal+os.sep+version+os.sep+"nonKP/"+f"{galaxy_type}_clustering.dat.fits",columns=load_columns,tabulatedbool=tabulatedbool)
    else:
        gal_tab = read_table(fpath_gal+os.sep+version+os.sep+f"{galaxy_type}_clustering.dat.fits",columns=load_columns,tabulatedbool=tabulatedbool)
    #gal_tab_SGC = read_table(fpath_gal+os.sep+version+os.sep+f"{galaxy_type}_SGC_clustering.dat.fits",columns=load_columns)
    #gal_tab = vstack((gal_tab_NGC, gal_tab_SGC))

    # apply the redshift cuts
    mask_zbins = np.ones(len(gal_tab),dtype=bool)
    if zmin is not None:
        mask_zbins &= (gal_tab['Z'] >= zmin)
    if zmax is not None:
        mask_zbins &= (gal_tab['Z'] < zmax)
    gal_tab = gal_tab[mask_zbins]
    
    # apply the photometric cuts (it is necessary since a few galaxies do not pass the initial photo-z cuts)
    # I am not sure why that is. It is only ~10 galaxies though, so the error should be irrelevant
    selection_mask = apply_photocuts_DESI(gal_tab,galaxy_type)
    magnitude_mask = apply_magnitude_cuts(gal_tab,galaxy_type,config)
    secondery_mask = apply_secondary_cuts(gal_tab,galaxy_type)
    if not np.all(secondery_mask):
        print("*"*50)
        raise ValueError(f"Secondary properties remove {np.sum(~secondery_mask)} galaxies! This should not happen.")

    print(f"Loaded {len(gal_tab)} {galaxy_type} galaxies, {len(gal_tab)-np.sum(selection_mask & magnitude_mask)} did not pass the photometric cuts")
    return gal_tab[selection_mask & magnitude_mask]



def apply_lensing(data,  kappa,  galaxy_type, config, verbose=False ):
    """Apply a small amount of lensing kappa to the observed magnitudes of the galaxy data. Combines all the functions to correctly apply the lensing for each type of magnitude in SDSS BOSS.

    Args:
        data_local : galaxy data
        kappa (float): lensing kappa
        galaxy_type: which galaxy sample

    Returns:
        data: copy of galaxy data with extra columns for lensed magnitudes named ..._mag
    """

    data_mag = copy.deepcopy(data)

    #note FLUX_IVAR_* are all only compared to >0. That can't be affected by lensing therefore can ignore
    # 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1'
    
    
    #for both LRG and BGS_BRIGHT need 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1'
    #note: want to apply lensing after dereddening but for flux deredenning is multiplicative just like lensing so they are interchangable.
    columns_to_magnify = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1']
    for column_to_magnify in columns_to_magnify:
        data_mag[column_to_magnify] *= (1.+2.*kappa)

    #absolute magnitude for BGS
    if(galaxy_type == "BGS_BRIGHT"):
        #absolute mag calculation commutes with additive change in the aparent magnitude calculation
        if "DA2" not in config["general"]["full_lss_path"]:
            data_mag["ABSMAG01_SDSS_R"] += - 2.5 * np.log10(1.+2.*kappa)
        #sign: for positive kappa galaxy gets brighter -> aparent magnitude gets smaller



    #the additional Fiber fluxes are more nuianced. Need size information for the galaxies to get an accurate estiamte,
    #e.g. a radius 

    if(galaxy_type == "LRG"):
        fiber_column = "FIBERFLUX_Z"
        fiber_tot_column = "FIBERTOTFLUX_Z"
    elif(galaxy_type == "BGS_BRIGHT"):
        fiber_column = "FIBERFLUX_R"
        fiber_tot_column = "FIBERTOTFLUX_R"
    elif(galaxy_type == 'ELG_LOPnotqso'):
        fiber_column = "FIBERFLUX_G"
    else:
        raise ValueError(f"galaxy_type {galaxy_type} not recognized")
    
    #only magnifying the galaxy not the surrounding light considered for FIBERTOTFLUX
    if galaxy_type != 'ELG_LOPnotqso':
        diff_fibertot_fiber = data_mag[fiber_tot_column] - data_mag[fiber_column]


    #fiber correction
    theta_e_galaxies = np.sqrt(data_mag["SHAPE_R"]**2 + 1) # half light radius in arcsec ## SDSS: data_local['R_DEV'] *  0.396 #convert pixel to arcsec
    theta_e_arr = np.arange(0.05, 10, 0.01)
    fiber_mag_lensing = config['general']['fiber_mag_lensing']
    if(fiber_mag_lensing == "DeVaucouleurs_profile"):
        cor_for_2fiber_mag_arr = [get_cor_for_1p5fiber_mag(theta_e=i, use_exp_profile=False) for i in theta_e_arr]
    elif(fiber_mag_lensing == "Exponential_profile"):
        cor_for_2fiber_mag_arr = [get_cor_for_1p5fiber_mag(theta_e=i, use_exp_profile=True) for i in theta_e_arr]
    elif(fiber_mag_lensing == "no_correction"):
        cor_for_2fiber_mag_arr = [0. for i in theta_e_arr]
    elif (fiber_mag_lensing == 'Tabulated'):
        fiber_correction = np.zeros_like(theta_e_galaxies)
        #fiber_correction = np.ones(len(cat['RA_1']))
        morphtypes = ['REX','DEV','EXP','SER','PSF']
        for morphtype in morphtypes:
            cat_sel = data_mag['MORPHTYPE'] == morphtype
            if morphtype == 'REX':
                dat = np.load('galaxy_fiber_info_files/rex.npz')
                fiber_correction[cat_sel] = np.interp(data_mag[cat_sel]['SHAPE_R'], dat['shape_r'], dat['f_factor'])
            elif morphtype == 'DEV':
                dat = np.load('galaxy_fiber_info_files/dev_fiber_factor.npz')
                spl = NearestNDInterpolator((dat['shape_r'],dat['q']),dat['f_factor'])
                fiber_correction[cat_sel] = spl((data_mag[cat_sel]['SHAPE_R'], data_mag[cat_sel]['AXIS_RATIO']))
            elif morphtype == 'EXP':
                dat = np.load('galaxy_fiber_info_files/exp_fiber_factor.npz')
                spl = NearestNDInterpolator((dat['shape_r'],dat['q']),dat['f_factor'])
                fiber_correction[cat_sel] = spl((data_mag[cat_sel]['SHAPE_R'], data_mag[cat_sel]['AXIS_RATIO']))
            elif morphtype == 'SER':
                #if sersic >= 2.5:
                dat1 = np.load('galaxy_fiber_info_files/dev_fiber_factor.npz')
                spl1 = NearestNDInterpolator((dat1['shape_r'],dat1['q']),dat1['f_factor'])
                dat2 = np.load('galaxy_fiber_info_files/exp_fiber_factor.npz')
                spl2 = NearestNDInterpolator((dat2['shape_r'],dat2['q']),dat2['f_factor'])

                fiber_correction[cat_sel & (data_mag['SERSIC'] >= 2.5)] = spl((data_mag[cat_sel & (data_mag['SERSIC'] >= 2.5)]['SHAPE_R'], data_mag['AXIS_RATIO'][cat_sel & (data_mag['SERSIC'] >= 2.5)]))
                #else:
                fiber_correction[cat_sel & (data_mag['SERSIC'] < 2.5)] = spl((data_mag[cat_sel & (data_mag['SERSIC'] < 2.5)]['SHAPE_R'], data_mag['AXIS_RATIO'][cat_sel & (data_mag['SERSIC'] < 2.5)])	)
            elif morphtype == 'PSF':
                fiber_correction[cat_sel] = 1

    else:
        raise ValueError(f"fiber_mag_lensing {fiber_mag_lensing} not recognized")

    if (fiber_mag_lensing != 'Tabulated'):
        fiber_correction = np.interp(theta_e_galaxies, theta_e_arr, cor_for_2fiber_mag_arr)
    
    
    # to lens secondary properties we need difference between unmagnified and magnified fiber flux
    if(config.getboolean("general","apply_cut_secondary_properties")):
        secondary_properties_fiber_column = config["secondary_properties"][f"Xval_{galaxy_type}"]
        fibermag_unmagnified = copy.deepcopy(data_mag[secondary_properties_fiber_column])

    # lensing the fiber fluxes
    data_mag[fiber_column] *= (1. +(2.- fiber_correction)*kappa)
    if galaxy_type != 'ELG_LOPnotqso':
        data_mag[fiber_tot_column] =  diff_fibertot_fiber + data_mag[fiber_column]

    #lensing the secondary properties
    if(config.getboolean("general","apply_cut_secondary_properties")):
        data_mag = apply_lensing_secondary_properties(data_mag, fibermag_unmagnified, galaxy_type, config, verbose=verbose)

    #If your survey only uses magnitudes that capture the full light of the galaxies, psf magnitudes and aperture magnitudes you can copy the method apply_lensing_v3 provided in magnification_bias_SDSS.py and just change the labels of the magnitudes used in your survey.
    #note has to work for negative kappa too!
    return data_mag

def apply_lensing_secondary_properties(data, fibermag_unmagnified, galaxy_type, config, verbose=False ):
    try:
        with open(os.path.dirname(os.path.abspath(__file__))+os.sep+"results"+os.sep+config["general"]["version"]+os.sep+"fit_results"+os.sep+"secondary_quantity_fits.json",'r') as f:
            fit_param_dict = json.load(f)
    except Exception as e:
        print("Error: {}".format(e))
        print("Error: Could not load the fit parameters for the secondary properties. Please run the secondary_cuts.py script first.")
        import sys
        sys.exit(1)

    secondary_properties_fiber_column = config["secondary_properties"][f"Xval_{galaxy_type}"]
    affected_secondary_properties = config["secondary_properties"][f"relevant_secondary_properties_{galaxy_type}"].strip().split(",")
    for secondary_property in affected_secondary_properties:
        a,b = fit_param_dict[f"{galaxy_type}_{secondary_properties_fiber_column}_{secondary_property}"]
        # do a 1st-order Taylor expansion of the fitted power-law a*x^b
        data[secondary_property] = data[secondary_property] + a*b*np.power(fibermag_unmagnified, b-1.)*(data[secondary_properties_fiber_column]-fibermag_unmagnified)
    return data

def get_weights(weights_str, data, galaxy_type):
    #implement the weights used for your galaxy survey. We used a string to switch between options but you can of course change that convention
    if weights_str == 'none':
        weights = None
    elif weights_str == 'weight_FKP':
        weights = data['WEIGHT']*data['WEIGHT_FKP']
    elif weights_str == 'weight':
        weights = data['WEIGHT']
    return weights


#helper functions for fiber flux magnification
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

def get_cor_for_1p5fiber_mag(theta_e, use_exp_profile = False):
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
    return np.interp(1.5 , fiber_sizes, dlnF_dlntheta )


#helper functions for alpha calculation
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


def apply_all_cuts(full_tab,galaxy_type,config, verbose = False):
    selection_mask = apply_photocuts_DESI(full_tab,galaxy_type)
    magnitude_mask = apply_magnitude_cuts(full_tab,galaxy_type,config)
    if(config.getboolean("general","apply_cut_secondary_properties")):
        secondery_mask = apply_secondary_cuts(full_tab,galaxy_type)
        if(verbose):
            print("Secondary properties remove {}/{} galaxies".format(np.sum(~secondery_mask), len(secondery_mask)))
    else:
        secondery_mask = np.ones(len(full_tab),dtype=bool)
    return selection_mask * magnitude_mask * secondery_mask

def apply_all_cuts_individual_cuts(full_tab,galaxy_type,config,verbose=False):
    masks_tab = apply_photocuts_DESI_individual_cuts(full_tab,galaxy_type)
    masks_tab.add_column(apply_magnitude_cuts(full_tab,galaxy_type,config),name="absolute magnitude cuts")
    if(config.getboolean("general","apply_cut_secondary_properties")):
        masks_tab = hstack([masks_tab,apply_secondary_cuts_individual_cuts(full_tab,galaxy_type)],join_type="exact")
        # if(verbose):
            # print("Secondary properties remove {}/{} galaxies".format(np.sum(~secondery_mask), len(secondery_mask)))
    return masks_tab



#calculate alpha from a single step size
def calculate_alpha_simple_DESI(data, kappa, galaxy_type, config, lensing_func =apply_lensing , weights_str="none"): 
    """Function to calculate the simple estimate for alpha for survey X

    Args:
        data : galaxy data
        kappa (float): kappa step size
        lensing_func (func, optional): Function to apply lensing to the data. Defaults to apply_lensing_v3.
        show_each_condition (bool, optional): Print out more details. Defaults to True.
        weights_str (str, optional): Weights for each galaxy. Defaults to "baseline".

    Returns:
        float: simple alpha estimate
        float: poisson error for alpha estimate
    """
    
    #assuming kappa positive
    weights = get_weights(weights_str, data, galaxy_type)
    #postivite kappa: increase #gal at faint end. 
    #convention: left-sided derivative on the faint end. So need minus sign
    data_mag = lensing_func(data,  kappa, galaxy_type, config)
    #print(5/0)

    combined_left = apply_all_cuts(data_mag, galaxy_type, config, verbose=True)
    
    print('sum of weights',np.sum(weights))
    print('kappa',kappa)
    print('Len of data_mag',np.shape(data_mag))
    print('Total combined_left',np.sum(combined_left))
    print('Sum of weights left',np.sum(weights[combined_left]))

    #other side
    data_mag = lensing_func(data,  -1.*kappa, galaxy_type, config)
    combined_right = apply_all_cuts(data_mag, galaxy_type, config, verbose=True)
    
    print('Total combined_right',np.sum(combined_right))
    print('Sum of weights right',np.sum(weights[combined_right]))


    
    alpha, alpha_error = get_alpha(combined_left, combined_right, kappa, weights=weights)
    print("-------")
    print("Overall alpha = {}".format(alpha))

    #redshift failurs currently not considered
    # R = magnification_bias_SDSS.get_R(data, use_exp_profile=use_exp_profile, case = "CMASS")
    # print("R = {} (not added)".format(R))

    return alpha, alpha_error

def calculate_alpha_simple_DESI_individual_cuts(data, kappa, galaxy_type, config, lensing_func =apply_lensing , weights_str="none"): 
    """Function to calculate the simple estimate for alpha for survey X

    Args:
        data : galaxy data
        kappa (float): kappa step size
        lensing_func (func, optional): Function to apply lensing to the data. Defaults to apply_lensing_v3.
        show_each_condition (bool, optional): Print out more details. Defaults to True.
        weights_str (str, optional): Weights for each galaxy. Defaults to "baseline".

    Returns:
        float: simple alpha estimate
        float: poisson error for alpha estimate
    """
    
    #assuming kappa positive
    weights = get_weights(weights_str, data, galaxy_type)
    #postivite kappa: increase #gal at faint end. 
    #convention: left-sided derivative on the faint end. So need minus sign
    data_mag = lensing_func(data,  kappa, galaxy_type, config)
    combined_left = apply_all_cuts_individual_cuts(data_mag, galaxy_type, config, verbose=True)

    if(config.getboolean("individual_cuts","validate_individual_cuts")):
        print("validating individual cuts left")
        combined_left_validation = apply_all_cuts(data_mag, galaxy_type, config, verbose=True)
        combined_left_sum = np.ones(len(combined_left),dtype=bool)
        for key in combined_left.colnames:
            combined_left_sum &= combined_left[key]
        assert np.all(combined_left_sum == combined_left_validation), "Error: individual cuts do not match the combined cuts"

    #other side
    data_mag = lensing_func(data,  -1.*kappa, galaxy_type, config)
    combined_right = apply_all_cuts_individual_cuts(data_mag, galaxy_type, config, verbose=True)
    if(config.getboolean("individual_cuts","validate_individual_cuts")):
        print("validating individual cuts right")
        combined_right_validation = apply_all_cuts(data_mag, galaxy_type, config, verbose=True)
        combined_right_sum = np.ones(len(combined_right),dtype=bool)
        for key in combined_right.colnames:
            combined_right_sum &= combined_right[key]
        assert np.all(combined_right_sum == combined_right_validation), "Error: individual cuts do not match the combined cuts"
    
    result_dict = {}
    n_gals = len(combined_left)
    for col in combined_left.colnames:
        alpha, alpha_error = get_alpha(combined_left[col], combined_right[col], kappa, weights=weights)
        result_dict[col] = [alpha, alpha_error, n_gals-np.sum(combined_left[col]), n_gals-np.sum(combined_right[col]), n_gals]
    print("-------")
    print("Overall alpha = {}".format(result_dict))

    #redshift failurs currently not considered
    # R = magnification_bias_SDSS.get_R(data, use_exp_profile=use_exp_profile, case = "CMASS")
    # print("R = {} (not added)".format(R))

    return result_dict


#calculate alpha from multiple step sizes
def calculate_alpha_DESI(data, kappas, galaxy_type, config, lensing_func =apply_lensing , weights_str="none"):  #, use_exp_profile=False
    """Baseline magnification bias estimate with our binwise estimator for CMASS.

    Args:
        data (_type_): galaxy data
        kappas (array): array of kappa steps
        lensing_func (func, optional): Function to apply lensing kappa to the data. Defaults to apply_lensing_v3.
        weights_str (array, optional): Weights for each galaxy. Using a string to select cases. Defaults to "baseline".
        #use_exp_profile (bool, optional): Switch to using an exponential profile. Defaults to False.

    Returns:
        dictionary: result for the magnification bias estimate
    """
    #assuming kappa positive
    weights = get_weights(weights_str, data, galaxy_type)
    #change in N for each kappa bin. 
    dNs = np.zeros_like(kappas)
    if(weights is None):
        N0 = len(data["RA"])
    else:
        N0 = np.sum(weights)
    kappa_step = kappas[1] - kappas[0] #assuming uniform spacing
    print("starting")

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
        data_mag = lensing_func(data,  kappa, galaxy_type, config)# use_exp_profile=use_exp_profile)
        combined_left = apply_all_cuts(data_mag, galaxy_type, config)
        
        #other side
        data_mag = lensing_func(data,  -1.*kappa, galaxy_type, config)# use_exp_profile=use_exp_profile)
        combined_right = apply_all_cuts(data_mag, galaxy_type, config)

        
        lst_left.append(combined_left)
        lst_right.append(combined_right)

    dNs, dNs_error, dNs_bins, dNs_bins_error = get_dN(lst_left, lst_right, weights=weights)

        
    result = {}
    result["dNs"] = dNs
    result["kappas"] = kappas
    result["N0"] = N0

    #dNs_bins = dNs[1:] - dNs[:-1]
    result["dNs_bins"] = dNs_bins

    norm = 1/(N0 * 2 * kappa_step)
    result["norm"] = norm
    result["As"]  = dNs_bins * norm

    result["As_error"] = dNs_bins_error*norm

    #getting the simple estimates for free
    #print("Not including R in the alpha simple estimate")
    result["alpha_simple"] = (dNs[1:] / (N0 * 2* kappas[1:])) #+R_over2N0
    result["alpha_simple_error"] = dNs_error[1:] / (N0 * 2* kappas[1:])
    result["alpha_simple_error_full"] = np.sqrt((dNs_error[1:]**2 / (N0 * 2* kappas[1:])**2) + (result["alpha_simple"]**2/N0))


    #linear fit
    xdats = result["kappas"][1:]-(kappa_step/2.)
    ydats = result["As"]
    sigmas = result["As_error"]
    result["fit"] = fit_linear(xdats, ydats, sigmas)

    
    return result


