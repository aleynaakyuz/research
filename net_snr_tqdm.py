import h5py
import numpy as np
import argparse
from pycbc.detector import Detector
from pycbc.waveform import get_fd_waveform
from pycbc.psd.read import from_txt
from pycbc.filter import sigma
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Path to injection files")
parser.add_argument("path", help="The string argument you want to pass")
parser.add_argument("det", nargs=2, help="detector names")
parser.add_argument("low_freq_cutoff", help="low_freq_cutoff")
parser.add_argument("df", help="delta_f")
parser.add_argument("psd_path", nargs=2, help="paths of the psd")


args = parser.parse_args()
input_path = args.path
det = args.det
psd_path = args.psd_path
low_freq = float(args.low_freq_cutoff)
df = float(args.df)

data = h5py.File(input_path, 'r')

def calculate_snr(det, path, ra, dec, pol, hp, hc):
    detec = Detector(det)
    f_plus, f_cross = detec.antenna_pattern(ra, dec, pol, t_gps=1697205750)
    proj_strain = f_plus*hp + f_cross*hc  
    psd = from_txt(path, low_freq_cutoff=low_freq, 
                    length=len(hp), delta_f=df)
    amp = sigma(proj_strain, psd=psd, low_frequency_cutoff=low_freq)

    return amp    


lenn = len(data['mass1'][:])

parameters = {
    "mass1": "mass1",
    "mass2": "mass2",
    "spin1x": "spin1x",
    "spin1y": "spin1y",
    "spin1z": "spin1z",
    "spin2x": "spin2x",
    "spin2y": "spin2y",
    "spin2z": "spin2z",
    "coa_phase": "coa_phase",
    "inclination": "inclination",
    "distance": "distance",
    "ra": "ra",
    "dec": "dec",
    "polarization": "polarization"
}

parameters2 = {
    "approximant": "IMRPhenomXPHM",
    "delta_f": df,
    "f_lower": low_freq
}

data_dic = {key: data[dataset_key][:] for key, dataset_key in parameters.items()}
temp_data = [{key: value[i] for key, value in data_dic.items()} for i in range(lenn)]
final_data = [{**temp_data[i], **parameters2} for i in range(lenn)]

hf = h5py.File('data_bbh_tqdm.h5', 'w')
for j in range(len(det)):
    net_snr_l = np.zeros(lenn)
    for k in tqdm(range(lenn)):
        for i in range(lenn):
            hp, hc = get_fd_waveform(**final_data[i])
            
            ra = final_data[i]['ra']
            dec = final_data[i]['dec']
            pol = final_data[i]['polarization']
            snr = calculate_snr(det[j], psd_path[j], ra, dec, pol, hp, hc)
            net_snr_l[i] = snr
    hf.create_dataset('network_snr' + str(det[j]), data=np.array(net_snr_l))
hf.close()
