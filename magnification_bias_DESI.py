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

def load_survey_data(galaxy_type,config,zmin=None,zmax=None,debug=True):
    fpath_lss = config['general']['full_lss_path']
    fpath_gal = config['general']['lensing_path']
    version = config['general']['version']

    required_columns = get_required_columns(galaxy_type)

    # load the LSS catalogue with all the columns we need
    lss_tab = Table(fitsio.read(fpath_lss+os.sep+version+os.sep+f"{galaxy_type}_full_HPmapcut.dat.fits",columns=required_columns))
    
    # load our catalogue that contains the clean sample
    gal_tab = Table(fitsio.read(fpath_gal+os.sep+version+os.sep+f"{galaxy_type}_full.dat.fits",columns=["TARGETID","Z"]))

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

    print(f"Loaded {len(full_tab)} {galaxy_type} galaxies, {len(full_tab)-np.sum(selection_mask)} did not pass the photometric cuts")
    return full_tab[selection_mask]

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
    return selection_mask


def apply_lensing(data_local,  kappa, plots=False, plot_output_filename="test.pdf", verbose=False, use_exp_profile=False):
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
    #If your survey only uses magnitudes that capture the full light of the galaxies, psf magnitudes and aperture magnitudes you can copy the method apply_lensing_v3 provided in magnification_bias_SDSS.py and just change the labels of the magnitudes used in your survey.
    #note has to work for negative kappa too!
    return None


def reapply_photocuts_surveyX(magnitudes_X, magnitudes_Y):
    #implement the full photometric selection of your survey and return a boolean array for which galaxies pass the selection
    return None

def get_weights(weights_str, data):
    #implement the weights used for your galaxy survey. We used a string to switch between options but you can of course change that convention
    weights = None 
    return weights


#calculate alpha from a single step size
def calculate_alpha_simple_surveyX(data, kappa, lensing_func =apply_lensing , weights_str="baseline", use_exp_profile=False): 
    import magnification_bias_SDSS
    """Function to calculate the simple estimate for alpha for survey X

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
    #TODO: fill in how to get the magnitudes from your data. E.g. magnitudes_X = data["magnitudes_X_mag"] . They need to be the magnified magnitudes!
    combined_left = reapply_photocuts_surveyX(TODO)
    
    #other side
    data = lensing_func(data,  -1.*kappa, use_exp_profile=use_exp_profile)
    combined_right = reapply_photocuts_surveyX(TODO)
    
    alpha, alpha_error = magnification_bias_SDSS.get_alpha(combined_left, combined_right, kappa, weights=weights)
    print("-------")
    print("Overall alpha without R= {}".format(alpha))

    R = magnification_bias_SDSS.get_R(data, use_exp_profile=use_exp_profile, case = "CMASS")
    print("R = {} (not added)".format(R))

    
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


