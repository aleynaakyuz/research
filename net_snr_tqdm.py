import h5py
import numpy as np
import argparse
from pycbc.detector import Detector
from pycbc.waveform import get_fd_waveform
from pycbc.psd.read import from_txt
from pycbc.filter import sigma
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Path to injection files")
parser.add_argument("path", help="Simulated population")
parser.add_argument("det", nargs=2, help="detector names")
parser.add_argument("low_freq_cutoff", help="low_freq_cutoff")
parser.add_argument("df_max", help="maximum delta_f")
parser.add_argument("psd_path", nargs=2, help="paths of the psd")
parser.add_argument("out_path", help="Output path")


args = parser.parse_args()
input_path = args.path
det = args.det
psd_path = args.psd_path
low_freq = float(args.low_freq_cutoff)
df_max = float(args.df_max)
out_path = args.out_path

data = h5py.File(input_path, 'r')

def calculate_snr(df, det, path, ra, dec, pol, hp, hc):
    detec = Detector(det)
    f_plus, f_cross = detec.antenna_pattern(ra, dec, pol, t_gps=1697205750)
    proj_strain = f_plus*hp + f_cross*hc  
    psd = from_txt(path, low_freq_cutoff=low_freq, 
                    length=len(hp), delta_f=df)
    amp = sigma(proj_strain, psd=psd, low_frequency_cutoff=low_freq)

    return amp    

def check_length(hp, hc, df):
    if len(hp) * df > 4000:
        inx = int(4000//df)
        hp = hp[:inx]
        hc = hc[:inx]
    return hp, hc

def opt_df(final_data, ra, dec, pol, det, path, snr_list):
    df = final_data['delta_f']
    hp, hc = get_fd_waveform(**final_data)
    hp, hc = check_length(hp, hc, df) 
    snr_l = calculate_snr(df, det, path, ra, dec, pol, hp, hc) 
    final_data.update({"delta_f": df/2})
    while df > 0.1:
        hp_s, hc_s = get_fd_waveform(**final_data)
        hp_s, hc_s = check_length(hp_s, hc_s, df) 
        snr_s = calculate_snr(df/2, det, path, ra, dec, pol, hp_s, hc_s)
        if abs(snr_l - snr_s) < 0.01:
            snr_list.append(snr_s)
            break
        else:
            df = df/2
            snr_l = snr_s
            final_data.update({"delta_f": df/2})
            continue


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
    "delta_f": df_max,
    "f_lower": low_freq
}

data_dic = {key: data[dataset_key][:] for key, dataset_key in parameters.items()}
temp_data = [{key: value[i] for key, value in data_dic.items()} for i in range(lenn)]
final_data = [{**temp_data[i], **parameters2} for i in range(lenn)]

hf = h5py.File(out_path, 'w')
for j in range(len(det)):
    snr_list = []
    for i in tqdm(range(lenn)):
        ra = final_data[i]['ra'] 
        dec = final_data[i]['dec'] 
        pol = final_data[i]['polarization'] 
        opt_df(final_data[i], ra, dec, pol, det[j], psd_path[j], snr_list)

    hf.create_dataset(str(det[j]), data=snr_list)
hf.close()
