import numpy as np
from pycbc.population.population_models import distance_from_rate, merger_rate_density, coalescence_rate, sfr_madau_dickinson_2014
from pycbc.distributions.utils import prior_from_config
import pycbc.workflow.configuration as wfc
import h5py

rho = 2*10**(-7)
z_max = 100
time = 5
file_name = 'parameters.h5'
path = ['/Users/aleyna/Pycbc/confusion-noise-3g-master-2/dataset_sim/population_files/prior_files/o1o2o3_lvk_bns_upper_5Hz.ini']
cp = wfc.WorkflowConfigParser(path)


joint_dist = prior_from_config(cp, prior_section='prior')
total_rate = joint_dist.bounds['total_rate'].min

merger_rate_dens = merger_rate_density(sfr_madau_dickinson_2014, 'inverse', rho_local=rho, maxz=z_max, npoints=1000)
coa_rate = coalescence_rate(merger_rate_dens, maxz=z_max, npoints=1000)
dist = distance_from_rate(total_rate, coa_rate)

samples = round(time * rho * dist**3)

data = joint_dist.rvs(samples)

datasets = list(data.fieldnames)
with h5py.File(file_name, "w") as file:
    for i in datasets:
        dataset = file.create_dataset(i, data=data[i])