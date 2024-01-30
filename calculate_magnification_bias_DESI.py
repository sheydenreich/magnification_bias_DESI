import configparser
import sys
#from magnification_bias_DESI import load_survey_data
import magnification_bias_DESI
import numpy as np

config = configparser.ConfigParser()
if len(sys.argv) > 1:
    config.read(sys.argv[1])
else:
    config.read('config.ini')

simple_alphas = {}

galaxy_types = config['general']['galaxy_types'].strip().split(',')
for galaxy_type in galaxy_types:
    print(f"Processing {galaxy_type}")
    z_bins = config['general']['zbins_'+galaxy_type].strip().split(',')
    z_bins = np.array(z_bins,dtype=float)

    simple_alphas_loc = np.zeros(len(z_bins)-1)
    for i in range(len(z_bins)-1):
        print(f'Processing bin {z_bins[i]} to {z_bins[i+1]}')
        galcat = magnification_bias_DESI.load_survey_data(galaxy_type,config,z_bins[i],z_bins[i+1])
        
        #reapply the photoz selection to check that there is no elements removed (or at least negligibly few)
        all_cond, each_condition = magnification_bias_DESI.reapply_photocuts_DESI
        #(cModelMag_new= data_CMASS["cModelMag_EC"], modelMag_new = data_CMASS["modelMag_EC"], fiber2Mag_new= data_CMASS["fiber2Mag_EC"], psfMag_new= data_CMASS["psfMag_EC"], dperp= data_CMASS["dperp"])
        print("Fraction of objects that fullfill baseline photometric selection for in {} = {}".format(i, np.sum(all_cond)/len(all_cond)))
        print(np.sum(all_cond)-len(all_cond))
        galcat = galcat[all_cond]

        #single step size
        simple_alphas_loc[i] = magnification_bias_DESI.calculate_alpha_simple_DESI(galcat, kappa=0.01)
        print("alpha = {}".format(simple_alphas_loc[i]))

        #TODO use full code with multiple step sizes
    simple_alphas[galaxy_type] = simple_alphas_loc
    


print("simple alpha results")
print(simple_alphas)