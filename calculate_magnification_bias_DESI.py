import configparser
import sys
#from magnification_bias_DESI import load_survey_data
import magnification_bias_DESI
import numpy as np
import json
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
config = configparser.ConfigParser()
if len(sys.argv) > 1:
    config.read(sys.argv[1])
else:
    config.read('config.ini')

simple_alphas = {}
alphas = {}
if config.getboolean('general','apply_individual_cuts'):
    alphas_individual_cuts = {}
do_full_alpha_stepwise_calculation = False
dkappa = 0.002
kappas = np.arange(0, 0.03+dkappa, dkappa)

# def convert_numpy_arrays_to_lists(data):
#     """
#     Recursively convert numpy arrays in a nested dictionary to lists.
    
#     :param data: The dictionary to convert.
#     :return: A new dictionary with numpy arrays converted to lists.
#     """
#     if isinstance(data, dict):
#         return {k: convert_numpy_arrays_to_lists(v) for k, v in data.items()}
#     elif isinstance(data, np.ndarray):
#         return data.tolist()
#     elif isinstance(data, list):
#         return [convert_numpy_arrays_to_lists(item) for item in data]
#     else:
#         return data

galaxy_types = config['general']['galaxy_types'].strip().split(',')
for galaxy_type in galaxy_types:
    print(f"Processing {galaxy_type}")
    z_bins = config['general']['zbins_'+galaxy_type].strip().split(',')
    z_bins = np.array(z_bins,dtype=float)

    simple_alphas_loc = np.zeros(len(z_bins)-1)
    simple_alphas_err_loc = np.zeros(len(z_bins)-1)
    alphas_loc = []
    for i in range(len(z_bins)-1):
        print(f'Processing bin {z_bins[i]} to {z_bins[i+1]}')
        galcat = magnification_bias_DESI.load_survey_data(galaxy_type,config,z_bins[i],z_bins[i+1])
        
        #(cModelMag_new= data_CMASS["cModelMag_EC"], modelMag_new = data_CMASS["modelMag_EC"], fiber2Mag_new= data_CMASS["fiber2Mag_EC"], psfMag_new= data_CMASS["psfMag_EC"], dperp= data_CMASS["dperp"])

        #single step size
        simple_alphas_loc[i],simple_alphas_err_loc[i] = magnification_bias_DESI.calculate_alpha_simple_DESI(galcat, kappa=0.01, galaxy_type=galaxy_type, config=config, weights_str="none")
        print("alpha_simple = {} +- {}".format(simple_alphas_loc[i],simple_alphas_err_loc[i]))

        if(config.getboolean('general','apply_individual_cuts')):
            print("Applying individual cuts")
            result_dict = magnification_bias_DESI.calculate_alpha_simple_DESI_individual_cuts(galcat, kappa=0.01, galaxy_type=galaxy_type, config=config, weights_str="none")
            alphas_individual_cuts[f"{galaxy_type}_{i}"] = result_dict

        if(do_full_alpha_stepwise_calculation):
            result = magnification_bias_DESI.calculate_alpha_DESI(galcat, kappas, galaxy_type=galaxy_type, config=config, weights_str="none")
            #print(result)
            print("alpha = {} +- {}".format(result["fit"]["alpha_fit"], result["fit"]["alpha_fit_error"]))
            alphas_loc.append(result)
    simple_alphas[galaxy_type] = list(simple_alphas_loc)
    simple_alphas[galaxy_type+"_error"] = list(simple_alphas_err_loc)
    alphas[galaxy_type] = alphas_loc
    

print("simple alpha results")
print(simple_alphas)

outpath = config['output']['output_path']
out_fname = config['output']['output_filename']

if(do_full_alpha_stepwise_calculation):
    print("full results")
    print(alphas)
    full_results = {"simple_alphas":simple_alphas, "alphas":alphas}
    full_out_fname = out_fname.split('.')[0]+'_full.json'
    os.makedirs(outpath,exist_ok=True)
    with open(outpath+full_out_fname, 'w', encoding='utf-8') as outfile:
        json.dump(full_results, outfile, ensure_ascii=False, indent=4, cls=NpEncoder)

    relevant_results = {}
    for galaxy_type in galaxy_types:
        relevant_results[galaxy_type] = {}
        relevant_results[galaxy_type]['simple_alphas'] = simple_alphas[galaxy_type]
        relevant_results[galaxy_type]['simple_alphas_error'] = simple_alphas[galaxy_type+'_error']
        relevant_results[galaxy_type]['alphas'] = [alphas[galaxy_type][i]['fit']['alpha_fit'] for i in range(len(z_bins)-1)]
        relevant_results[galaxy_type]['alphas_error'] = [alphas[galaxy_type][i]['fit']['alpha_fit_error'] for i in range(len(z_bins)-1)]

    with open(outpath+out_fname, 'w', encoding='utf-8') as outfile:
        json.dump(relevant_results, outfile, ensure_ascii=False, indent=4, cls=NpEncoder)

else:
    simple_alphas_fname = out_fname.split('.')[0]+'_simple.json'
    with open(outpath+simple_alphas_fname, 'w', encoding='utf-8') as outfile:
        json.dump(simple_alphas, outfile, ensure_ascii=False, indent=4, cls=NpEncoder)

if(config.getboolean('general','apply_individual_cuts')):
    individual_cuts_out_fname = out_fname.split('.')[0]+'_individual_cuts.json'
    with open(outpath+individual_cuts_out_fname, 'w', encoding='utf-8') as outfile:
        json.dump(alphas_individual_cuts, outfile, ensure_ascii=False, indent=4, cls=NpEncoder)