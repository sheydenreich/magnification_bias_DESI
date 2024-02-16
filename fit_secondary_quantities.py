from scipy.optimize import curve_fit
import numpy as np  
from  magnification_bias_DESI import get_required_columns,apply_magnitude_cuts
from astropy.table import Table
import fitsio
import os
import json


def power_law(x, a, b):
    return a * np.power(x, b)

def fit_power_law(xdata, ydata):
    # Fit the power-law model to the data
    params, covariance = curve_fit(power_law, xdata, ydata, sigma=np.sqrt(ydata), absolute_sigma=True)

    return params

def using_mpl_scatter_density(fig, x, y, cbar_label=None):
    import mpl_scatter_density
    from matplotlib.colors import LinearSegmentedColormap
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label=cbar_label)
    return ax

def fit_secondary_quantities(config):
    # some imports and definitions for the consistency plots
    import matplotlib.pyplot as plt

    plots_path = os.path.dirname(os.path.abspath(__file__))+os.sep+"results"+os.sep+config["general"]["version"]+os.sep+"validation_plots"+os.sep


    galaxy_types = config['general']['galaxy_types'].strip().split(',')
    fpath_lss = config['general']['full_lss_path']
    fpath_gal = config['general']['lensing_path']
    version = config['general']['version']

    secondary_quantity_dict = {}
    for galaxy_type in galaxy_types:
        required_columns = get_required_columns(galaxy_type)
        if galaxy_type[:3] in ["LRG","ELG"]:
            required_columns += ["Z_not4clus"]
        elif galaxy_type[:3] == "BGS":
            required_columns += ["Z_not4clus","ABSMAG01_SDSS_R"]

        # load the LSS catalogue with all the columns we need
        lss_tab = Table(fitsio.read(fpath_lss+os.sep+version+os.sep+f"{galaxy_type}_full_HPmapcut.dat.fits",columns=required_columns))
        
        # cut to the galaxy sample that is relevant for us
        # IMPORTANT: We do not apply the secondary cuts here, as they would otherwise bias the power-law fits
        magnitude_mask = apply_magnitude_cuts(lss_tab, galaxy_type, config, mag_col="ABSMAG01_SDSS_R", zcol="Z_not4clus")
        lss_tab = lss_tab[magnitude_mask]

        fit_xval = config['secondary_properties'][f"Xval_{galaxy_type}"]
        fit_yvals = config['secondary_properties'][f"relevant_secondary_properties_{galaxy_type}"].strip().split(',')

        for fit_yval in fit_yvals:
            mask = np.isfinite(lss_tab[fit_xval]) & np.isfinite(lss_tab[fit_yval])

            xdata = lss_tab[fit_xval][mask]
            ydata = lss_tab[fit_yval][mask]

            mask = (xdata > 0) & (ydata > 0)
            if not np.all(mask):
                print(f"Warning: {np.sum(~mask)} of {len(mask)} points are not positive, so they are not considered.")

            xdata = xdata[mask]
            ydata = ydata[mask]

            params = fit_power_law(xdata,ydata)

            print(f"Fitted Parameters for {galaxy_type}, {fit_xval} x {fit_yval}: {params}")

            os.makedirs(plots_path,exist_ok=True)

            # Plotting the data and the fitted curve
            fig = plt.figure()
            ax = using_mpl_scatter_density(fig, xdata, ydata)

            xarr = np.geomspace(np.min(xdata),np.max(xdata),100)
            plt.plot(xarr, power_law(xarr, *params), label='Fit')
            plt.xlabel(fit_xval)
            plt.ylabel(fit_yval)
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(plots_path+f"{galaxy_type}_{fit_xval}_vs_{fit_yval}.png",dpi=300,bbox_inches='tight')
            plt.close()

            secondary_quantity_dict[f"{galaxy_type}_{fit_xval}_{fit_yval}"] = list(params)
    os.makedirs(os.path.dirname(os.path.abspath(__file__))+os.sep+"results"+os.sep+config["general"]["version"]+os.sep+"fit_results"+os.sep,exist_ok=True)
    with open(os.path.dirname(os.path.abspath(__file__))+os.sep+"results"+os.sep+config["general"]["version"]+os.sep+"fit_results"+os.sep+"secondary_quantity_fits.json", 'w', encoding='utf-8') as f:
        json.dump(secondary_quantity_dict,f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import configparser
    import sys
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        config.read(sys.argv[1])
    else:
        config.read('config.ini')
    fit_secondary_quantities(config)
