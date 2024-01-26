import configparser
import sys
from magnification_bias_DESI import load_survey_data
import numpy as np

config = configparser.ConfigParser()
if len(sys.argv) > 1:
    config.read(sys.argv[1])
else:
    config.read('config.ini')

galaxy_types = config['general']['galaxy_types'].strip().split(',')
for galaxy_type in galaxy_types:
    print(f"Processing {galaxy_type}")
    z_bins = config['general']['zbins_'+galaxy_type].strip().split(',')
    z_bins = np.array(z_bins,dtype=float)
    for i in range(len(z_bins)-1):
        print(f'Processing bin {z_bins[i]} to {z_bins[i+1]}')
        galcat = load_survey_data(galaxy_type,config,z_bins[i],z_bins[i+1])