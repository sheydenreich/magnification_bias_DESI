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
alphas = {}
do_full_alpha_stepwise_calculation = True
dkappa = 0.002
kappas = np.arange(0, 0.03+dkappa, dkappa)

galaxy_types = config['general']['galaxy_types'].strip().split(',')
for galaxy_type in galaxy_types:
    print(f"Processing {galaxy_type}")
    z_bins = config['general']['zbins_'+galaxy_type].strip().split(',')
    z_bins = np.array(z_bins,dtype=float)

    simple_alphas_loc = np.zeros(len(z_bins)-1)
    alphas_loc = []
    for i in range(len(z_bins)-1):
        print(f'Processing bin {z_bins[i]} to {z_bins[i+1]}')
        galcat = magnification_bias_DESI.load_survey_data(galaxy_type,config,z_bins[i],z_bins[i+1])
        
        #(cModelMag_new= data_CMASS["cModelMag_EC"], modelMag_new = data_CMASS["modelMag_EC"], fiber2Mag_new= data_CMASS["fiber2Mag_EC"], psfMag_new= data_CMASS["psfMag_EC"], dperp= data_CMASS["dperp"])

        #single step size
        simple_alphas_loc[i],_ = magnification_bias_DESI.calculate_alpha_simple_DESI(galcat, kappa=0.01, galaxy_type=galaxy_type, weights_str="none")
        print("alpha_simple = {}".format(simple_alphas_loc[i]))

        if(do_full_alpha_stepwise_calculation):
            result = magnification_bias_DESI.calculate_alpha_DESI(galcat, kappas, galaxy_type=galaxy_type, weights_str="none")
            #print(result)
            print("alpha = {} +- {}".format(result["fit"]["alpha_fit"], result["fit"]["alpha_fit_error"]))
            alphas_loc.append(result)
    simple_alphas[galaxy_type] = simple_alphas_loc
    alphas[galaxy_type] = alphas_loc
    


print("simple alpha results")
print(simple_alphas)

print("full results")
print(alphas)

#TODO use full code with multiple step sizes decide how to save the information