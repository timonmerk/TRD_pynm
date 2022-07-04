import pandas as pd
import numpy as np
import os

from py_neuromodulation import nm_RMAP, nm_plots

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

df = pd.read_pickle("df_fft_sw.p")

df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
df_ind_ch["sub_str"] = df_ind_ch["sub"].str.split("_").apply(lambda x: x[1])

rmap = nm_RMAP.RMAPChannelSelector()

# Choose functional or structural connectivity
# Note: functional connectivity requires the cond_str "_AvgR_Fz.nii"
# while for the structural connectivity cond_str can be None

path_dir = r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\Analysis connectomics\func_conn"
path_dir = r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\Analysis connectomics\str_conn"

path_out = (
    r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics"
)

l_fps_names, l_fps_dat = rmap.load_all_fingerprints(
    path_dir, cond_str="_AvgR_Fz.nii"
)

l_fps_names, l_fps_dat = rmap.load_all_fingerprints(path_dir, cond_str=None)

conn_arr, l_per = get_performances_and_flattened_fingerprints(l_fps_names, df_ind_ch)
rmap_arr_np = np.nan_to_num(rmap.get_RMAP(np.array(conn_arr).T, np.array(l_per)))

rmap.save_Nii(rmap_arr_np, rmap.affine, name=os.path.join(path_out, "rmap.nii"))


# leave one ch. out Cross Validation
per_left_out, per_predict = rmap.leave_one_ch_out_cv(
    l_fps_names, l_fps_dat, l_per
)

nm_plots.reg_plot(
    x_col="test_performance", y_col="struct_conn_predict", data=df_plt_corr
)

subjects = list(df_ind_ch["sub_str"].unique())

per_left_out, per_predict = rmap.leave_one_sub_out_cv(
    l_fps_names, l_fps_dat, l_per, subjects
)


