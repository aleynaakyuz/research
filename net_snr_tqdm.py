import h5py
import argparse
from pycbc.detector import Detector
from pycbc.waveform import get_fd_waveform
from pycbc.psd.read import from_txt
from pycbc.filter import sigma
from tqdm import tqdm
import numpy as np

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
df_min = 0.125
f_max = 4000


data = h5py.File(input_path, 'r')

def read_psds(psd_path, df_max, df_min, f_max): 
    dfs = [df_max]
    
    while dfs[-1] > df_min:
        dfs.append(dfs[-1]/2)
    df_arr = np.array(dfs)
    
    psds = {}
    for i in df_arr:
        lenn = int(f_max / i) 
        psd = from_txt(psd_path, low_freq_cutoff=low_freq, 
                        length=lenn, delta_f=i)
        psds.update({i: psd})
    return psds

def calculate_snr(df, det, psd, ra, dec, pol, hp, hc):
    detec = Detector(det)
    f_plus, f_cross = detec.antenna_pattern(ra, dec, pol, t_gps=1697205750)
    proj_strain = f_plus*hp + f_cross*hc  
    amp = sigma(proj_strain, psd=psd[df], low_frequency_cutoff=low_freq)

    return amp    

def check_length(hp, hc, df):
    if len(hp) * df > f_max:
        inx = int(f_max//df)
        hp = hp[:inx]
        hc = hc[:inx]
    return hp, hc

def opt_df(final_data, ra, dec, pol, det, psd):
    df = final_data['delta_f']
    hp, hc = get_fd_waveform(**final_data)
    hp, hc = check_length(hp, hc, df) 
    snr_l = calculate_snr(df, det, psd, ra, dec, pol, hp, hc) 
    final_data.update({"delta_f": df/2})
    while df > df_min:
        hp_s, hc_s = get_fd_waveform(**final_data)
        hp_s, hc_s = check_length(hp_s, hc_s, df/2) 
        snr_s = calculate_snr(df/2, det, psd, ra, dec, pol, hp_s, hc_s)
        if abs(snr_l - snr_s)/snr_l < 0.01:
            break
        else:
            df = df/2
            snr_l = snr_s
            final_data.update({"delta_f": df/2})
            continue
    return snr_l


lenn = len(data['mass1'][:])

temp_params = list(data.keys())
parameters = {key: key for key in temp_params}

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
    psd = read_psds(psd_path[j], df_max, df_min, f_max)
    for i in tqdm(range(lenn)):
        ra = final_data[i]['ra'] 
        dec = final_data[i]['dec'] 
        pol = final_data[i]['polarization'] 
        snr = opt_df(final_data[i], ra, dec, pol, det[j], psd)
        snr_list.append(snr)

    hf.create_dataset(str(det[j]), data=snr_list)
hf.close()