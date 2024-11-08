import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sb
from joblib import Parallel, delayed
import pickle
from scipy import stats

from py_neuromodulation import nm_RMAP, nm_plots, nm_stats

def get_performances_and_flattened_fingerprints(l_fps_names: list, df_ind_ch: pd.DataFrame):

    l_per = []
    conn_arr = []
    for idx, f in enumerate(l_fps_names):
        sub = f.split("_")[0][4:]  # are used for the query below
        ch = f.split("_")[2]

        l_per.append(
            df_ind_ch.query("ch == @ch and sub_str == @sub").iloc[0][
                "performance_test"
            ]
        )

        conn_arr.append(np.nan_to_num(l_fps_dat[idx].flatten()))
    return conn_arr, l_per

df = pd.read_pickle("scripts\\df_sw_fft_ntr_vs_plsunpls.p")

df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
df_ind_ch["sub_str"] = df_ind_ch["sub"].str.split("_").apply(lambda x: x[1])

rmap = nm_RMAP.RMAPChannelSelector()

# Choose functional or structural connectivity
# Note: functional connectivity requires the cond_str "_AvgR_Fz.nii"
# while for the structural connectivity cond_str can be None

path_out = (
    r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics"
)

path_dir = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics\conn_out\str_conn_smooth_888" # str_conn_smooth

PATH_RMAP_IRMEN = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\Paper\Spatial correlation Emotion R-Maps\smooth_fi.nii"

PATH_RMAP_TIMON = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\Paper\Spatial correlation Emotion R-Maps\smooth_tm.nii"

rmap_timon = rmap.load_fingerprint(PATH_RMAP_TIMON).flatten()

rmap_irmen = rmap.load_fingerprint(PATH_RMAP_IRMEN).flatten()

rmap_irmen[np.where(rmap_irmen == 0)] = np.nan
idx_not_none_irmen = np.where(np.logical_not(np.isnan(rmap_irmen)))[0]

idx_not_none = np.where(np.logical_not(np.isnan(rmap_timon)))[0]
idx_sel = list(set(idx_not_none_irmen).intersection(idx_not_none))


r_orig = stats.spearmanr(rmap_irmen[idx_sel], rmap_timon[idx_sel])[0]
r_orig = stats.spearmanr(rmap_irmen, rmap_timon)[0]


# problem sind die nullen, die Korrelation funktioniert direkt, wenn man die nullen, gelesen mit smooth_fi und smooth_tm gleich mit einliesst

rho_orig = np.corrcoef(rmap_irmen[idx_sel], rmap_timon[idx_sel])[0, 1]




l_fps_names, l_fps_dat = rmap.load_all_fingerprints(path_dir, cond_str=None)
l_fps_names = [l[1:] for l in l_fps_names]  # remove s for smoothed images
rmap_name = "rmap_str_smoothed_888.nii"  # 
str_add = "str"

conn_arr, l_per = get_performances_and_flattened_fingerprints(l_fps_names, df_ind_ch)
rmap_orig = np.nan_to_num(rmap.get_RMAP(np.array(conn_arr).T, np.array(l_per)))

rmap_orig[np.where(rmap_orig == 0)] = np.nan

corr_val_orig = rmap.get_corr_numba(np.array(rmap_irmen), rmap_orig)

l_shuffled_corr = []
l_shuffled_rmap = []

arr_per = np.array(l_per)

for i in range(1000):
    print(i)
    np.random.shuffle(arr_per)
    rmap_arr_np = np.nan_to_num(rmap.get_RMAP(np.array(conn_arr).T, arr_per))
    rmap_arr_np[np.where(rmap_arr_np == 0)] = np.nan

    #rmap.save_Nii(rmap_arr_np, rmap.affine, name=os.path.join(path_out, rmap_name))

    # for every R-MAP run a correlation, save them in a list, run a permutation test
    l_shuffled_rmap.append(rmap_arr_np)


with open("l_fingerprints.p", "wb") as fp:   #Pickling
   pickle.dump(l_shuffled_rmap, fp)

stat_rho_l = []

rmap_irmen[np.where(rmap_irmen == 0)] = np.nan
idx_not_none_irmen = np.where(np.logical_not(np.isnan(rmap_irmen)))[0]

stat_r_l = []

for i in range(1000):
    print(i)
    idx_not_none = np.where(np.logical_not(np.isnan(l_shuffled_rmap[i])))[0]

    idx_sel = list(set(idx_not_none_irmen).intersection(idx_not_none))

    #stat_rho_l.append(
    #    stats.spearmanr(rmap_irmen[idx_sel], np.array(l_shuffled_rmap[i][idx_sel]))[0]
    #)

    stat_r_l.append(np.corrcoef(rmap_irmen[idx_sel], np.array(l_shuffled_rmap[i][idx_sel]))[0, 1])

# compare the correlation of the original one:
idx_not_none = np.where(np.logical_not(np.isnan(rmap_orig)))[0]
idx_sel = list(set(idx_not_none_irmen).intersection(idx_not_none))
r_orig = stats.spearmanr(rmap_irmen[idx_sel], rmap_orig[idx_sel])[0]
rho_orig = np.corrcoef(rmap_irmen[idx_sel], rmap_orig[idx_sel])[0, 1]

z_ch, p_ch = nm_stats.permutation_numba_onesample(
    np.array(stat_r_l),
    r_orig,
    5000,
    False,
)

# next step: calculate the correlation values

#corr_val = rmap.get_corr_numba(rmap_irmen, rmap_arr_np)


idx_not_none = np.where(np.logical_not(np.isnan(rmap_timon)))[0]
idx_sel = list(set(idx_not_none_irmen).intersection(idx_not_none))
r_orig = stats.spearmanr(rmap_irmen[idx_sel], rmap_timon[idx_sel])[0]
rho_orig = np.corrcoef(rmap_irmen[idx_sel], rmap_timon[idx_sel])[0, 1]




parallel = Parallel(n_jobs=59)
output_generator = parallel(delayed(get_shuffled_per)() for _ in range(1000))
print(list(output_generator))

z_ch, p_ch = nm_stats.permutation_numba_onesample(
    np.array(output_generator),
    corr_val_orig,
    5000,
    False,
)
