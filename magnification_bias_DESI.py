# To apply the magnification bias estimate to another survey you need:
# * The galaxy catalog including the magnitudes used for the photometric selection
# * The exact conditions used for the photometric selection
# * An understanding of how the magnitudes used behave under lensing. In our work for SDSS BOSS we characterized this for magnitudes that capture the full light of the galaxy, psf magnitudes and aperture magnitudes. If you need other magnitudes you need to characterize them yourself.

#Below we lay out empty functions you need to fill in for your specific survey. You essentially need to fill out one function for each of the bullet points above.

#For an example see magnification_bias_SDSS.py

import numpy as np
from istarget import select_lrg,select_bgs_bright,get_required_columns
from astropy.io import fits
from astropy.table import Table,join
import os
import fitsio

import copy

def get_redshift_bins(galaxy_type):
    if(galaxy_type == "LRG"):
        return np.array([0.4,0.6,0.8,1.1])
    elif(galaxy_type=="BGS_BRIGHT"):
        return np.array([0.1,0.2,0.3,0.4])
    else:
        raise ValueError("Invalid value of galaxy_type in get_redshift_bins. Allowed: [BGS_BRIGHT,LRG]. Here: {}".format(galaxy_type))

def get_magnitude_cuts(galaxy_type):
    if(galaxy_type in ["LRG","ELG"]):
        return None
    elif(galaxy_type=="BGS_BRIGHT"):
        return -1.*np.array([19.5,20.5,21.0])
    else:
        raise ValueError("Invalid value of galaxy_type in get_magnitude_cuts. Allowed: [BGS_BRIGHT,LRG]. Here: {}".format(galaxy_type))

def create_redshift_mask(reference_redshifts,z_bins_lens):
    if z_bins_lens is None:
        return np.ones(len(reference_redshifts),dtype=bool)
    else:
        redshift_mask = ((reference_redshifts >= z_bins_lens[0]) & (reference_redshifts < z_bins_lens[-1]))
        return redshift_mask


def get_magnitude_mask(data_table,magnitude_cuts,lens_bins,mag_col="ABSMAG_RP0",zcol="Z"):
    if magnitude_cuts is None:
        return np.ones(len(data_table),dtype=bool)
    redshift_mask = create_redshift_mask(data_table[zcol],lens_bins)
    lens_zbins = (np.digitize(data_table[zcol],lens_bins)-1).astype(int)
    mask_magnitudes = np.zeros(len(data_table),dtype=bool)
    mask_magnitudes[redshift_mask] = (data_table[mag_col][redshift_mask] < magnitude_cuts[lens_zbins[redshift_mask]])
    return mask_magnitudes

def apply_magnitude_cuts(data_table,galaxy_type):
    magnitude_cuts = get_magnitude_cuts(galaxy_type)
    lens_bins = get_redshift_bins(galaxy_type)
    mask_magnitudes = get_magnitude_mask(data_table,magnitude_cuts,lens_bins)
    return mask_magnitudes

def load_survey_data(galaxy_type,config,zmin=None,zmax=None,debug=True):
    fpath_lss = config['general']['full_lss_path']
    fpath_gal = config['general']['lensing_path']
    version = config['general']['version']

    required_columns = get_required_columns(galaxy_type)

    # load the LSS catalogue with all the columns we need
    lss_tab = Table(fitsio.read(fpath_lss+os.sep+version+os.sep+f"{galaxy_type}_full_HPmapcut.dat.fits",columns=required_columns))
    
    # load our catalogue that contains the clean sample
    load_columns = ["TARGETID","Z"]
    if galaxy_type == "BGS_BRIGHT":
        load_columns += ["ABSMAG_RP0"]
    gal_tab = Table(fitsio.read(fpath_gal+os.sep+version+os.sep+f"{galaxy_type}_full.dat.fits",columns=load_columns))

    if(debug):
        # Extract TARGETID columns
        targetids_gal_tab = set(gal_tab['TARGETID'])
        targetids_lss_tab = set(lss_tab['TARGETID'])

        # Check if every TARGETID in gal_tab is in lss_tab
        missing_targetids = targetids_gal_tab - targetids_lss_tab

        if missing_targetids:
            print(f"Missing TARGETID(s) in lss_tab: {missing_targetids}")
        else:
            print("All TARGETID(s) in gal_tab are present in lss_tab")

    # cut the full LSS catalogue to the clean sample
    full_tab = join(gal_tab,lss_tab,keys='TARGETID',join_type='left')

    # apply the redshift cuts
    mask_zbins = np.ones(len(full_tab),dtype=bool)
    if zmin is not None:
        mask_zbins &= (full_tab['Z'] >= zmin)
    if zmax is not None:
        mask_zbins &= (full_tab['Z'] < zmax)
    full_tab = full_tab[mask_zbins]
    
    # apply the photometric cuts (it is necessary since a few galaxies do not pass the initial photo-z cuts)
    # I am not sure why that is. It is only ~10 galaxies though, so the error should be irrelevant
    selection_mask = apply_photocuts_DESI(full_tab,galaxy_type)
    magnitude_mask = apply_magnitude_cuts(full_tab,galaxy_type)

    print(f"Loaded {len(full_tab)} {galaxy_type} galaxies, {len(full_tab)-np.sum(selection_mask & magnitude_mask)} did not pass the photometric cuts")
    return full_tab[selection_mask & magnitude_mask]

def apply_photocuts_DESI(data, galaxy_type):
    if galaxy_type == "LRG":
        selection_fnc = select_lrg
    elif galaxy_type == "BGS_BRIGHT":
        selection_fnc = select_bgs_bright
    else:
        raise ValueError(f"galaxy_type {galaxy_type} not recognized")
    # split into north and south region, as selection function is different
    mask_north = (data['PHOTSYS'] == 'N')
    photoz_selection_north = selection_fnc(data[mask_north],field='north')
    photoz_selection_south = selection_fnc(data[~mask_north],field='south')
    selection_mask = np.zeros(len(data),dtype=bool)
    selection_mask[mask_north] = photoz_selection_north
    selection_mask[~mask_north] = photoz_selection_south
    magnitude_mask = apply_magnitude_cuts(data,galaxy_type)
    return (selection_mask & magnitude_mask)


def apply_lensing(data,  kappa,  galaxy_type, verbose=False ):
    """Apply a small amount of lensing kappa to the observed magnitudes of the galaxy data. Combines all the functions to correctly apply the lensing for each type of magnitude in SDSS BOSS.

    Args:
        data_local : galaxy data
        kappa (float): lensing kappa
        galaxy_type: which galaxy sample

    Returns:
        data: copy of galaxy data with extra columns for lensed magnitudes named ..._mag
    """
    #TODO test if this works for the data_mag object. Want to copy it instead of overwriting the values
    #having seperate columns for the magnified fluxes would be more memory efficient but that would requires significant
    #changes in istarget.py
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
        data_mag["ABSMAG_RP0"] += - 2.5 * np.log10(1.+2.*kappa)
        #sign: for positive kappa galaxy gets brighter -> aparent magnitude gets smaller



    #the additional Fiber fluxes are more nuianced. Need size information for the galaxies to get an accurate estiamte,
    #e.g. a radius 

    if(galaxy_type == "LRG"):
        fiber_column = "FIBERFLUX_Z"
        fiber_tot_column = "FIBERTOTFLUX_Z"
    elif(galaxy_type == "BGS_BRIGHT"):
        fiber_column = "FIBERFLUX_R"
        fiber_tot_column = "FIBERTOTFLUX_R"
    else:
        raise ValueError(f"galaxy_type {galaxy_type} not recognized")
    
    #only magnifying the galaxy not the surrounding light considered for FIBERTOTFLUX
    diff_fibertot_fiber = data_mag[fiber_tot_column] - data_mag[fiber_column]


    #fiber correction
    theta_e_galaxies = data_mag["SHAPE_R"] # half light radius in arcsec ## SDSS: data_local['R_DEV'] *  0.396 #convert pixel to arcsec
    theta_e_arr = np.arange(0.05, 10, 0.01)
    cor_for_2fiber_mag_arr = [get_cor_for_2fiber_mag(theta_e=i, use_exp_profile=False) for i in theta_e_arr]
    fiber_correction = np.interp(theta_e_galaxies, theta_e_arr, cor_for_2fiber_mag_arr)
    #SDSS: data_local["fiber2Flux_mag"] = data_local["FIBER2FLUX"] * (1. +(2.- fiber_correction)*kappa)
    
    data_mag[fiber_column] *= (1. +(2.- fiber_correction)*kappa)


    data_mag[fiber_tot_column] =  diff_fibertot_fiber + data_mag[fiber_column]

    #If your survey only uses magnitudes that capture the full light of the galaxies, psf magnitudes and aperture magnitudes you can copy the method apply_lensing_v3 provided in magnification_bias_SDSS.py and just change the labels of the magnitudes used in your survey.
    #note has to work for negative kappa too!
    return data_mag


def get_weights(weights_str, data, galaxy_type):
    #implement the weights used for your galaxy survey. We used a string to switch between options but you can of course change that convention
    weights = None 
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


#calculate alpha from a single step size
def calculate_alpha_simple_DESI(data, kappa, galaxy_type, lensing_func =apply_lensing , weights_str="none"): 
    import magnification_bias_SDSS
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
    data_mag = lensing_func(data,  kappa, galaxy_type)

    combined_left = apply_photocuts_DESI(data_mag, galaxy_type)
    
    #other side
    data_mag = lensing_func(data,  -1.*kappa, galaxy_type)
    combined_right = apply_photocuts_DESI(data_mag, galaxy_type)
    
    alpha, alpha_error = get_alpha(combined_left, combined_right, kappa, weights=weights)
    print("-------")
    print("Overall alpha = {}".format(alpha))

    #redshift failurs currently not considered
    # R = magnification_bias_SDSS.get_R(data, use_exp_profile=use_exp_profile, case = "CMASS")
    # print("R = {} (not added)".format(R))

    return alpha, alpha_error

#calculate alpha from multiple step sizes
def calculate_alpha_surveyX(data, kappas, lensing_func =apply_lensing , weights_str="baseline", use_exp_profile=False): 
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
        N0 = len(data["Z"])
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
        combined_left, each_condition_left = reapply_photocuts_surveyX(TODO)
        
        #other side
        data = lensing_func(data,  -1.*kappa, use_exp_profile=use_exp_profile)
        combined_right, each_condition_right = reapply_photocuts_surveyX(TODO)
        
        lst_left.append(combined_left)
        lst_right.append(combined_right)

    dNs, dNs_error, dNs_bins, dNs_bins_error = magnification_bias_SDSS.get_dN(lst_left, lst_right, weights=weights)

        
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
    print("Not including R in the alpha simple estimate")
    result["alpha_simple"] = (dNs[1:] / (N0 * 2* kappas[1:])) #+R_over2N0
    result["alpha_simple_error"] = dNs_error[1:] / (N0 * 2* kappas[1:])
    result["alpha_simple_error_full"] = np.sqrt((dNs_error[1:]**2 / (N0 * 2* kappas[1:])**2) + (result["alpha_simple"]**2/N0))


    #linear fit
    xdats = result["kappas"][1:]-(kappa_step/2.)
    ydats = result["As"]
    sigmas = result["As_error"]
    result["fit"] = magnification_bias_SDSS.fit_linear(xdats, ydats, sigmas)

    
    return result


