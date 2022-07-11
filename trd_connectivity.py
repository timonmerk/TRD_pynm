import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sb

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

df = pd.read_pickle("df_sw_fft_ntr_vs_plsunpls.p")

df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
df_ind_ch["sub_str"] = df_ind_ch["sub"].str.split("_").apply(lambda x: x[1])

rmap = nm_RMAP.RMAPChannelSelector()

# Choose functional or structural connectivity
# Note: functional connectivity requires the cond_str "_AvgR_Fz.nii"
# while for the structural connectivity cond_str can be None

path_out = (
    r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics"
)

df_plt_corr = pd.DataFrame()
for run_func in [False, True]:
    if run_func is True:
        path_dir = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics\new_conn_out\func_conn"
        l_fps_names, l_fps_dat = rmap.load_all_fingerprints(
            path_dir, cond_str="_AvgR_Fz.nii"
        )
        rmap_name = "rmap_func.nii"
        str_add = "func"
    else:
        path_dir = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics\new_conn_out\str_conn_smooth_888" # str_conn_smooth

        l_fps_names, l_fps_dat = rmap.load_all_fingerprints(path_dir, cond_str=None)
        l_fps_names = [l[1:] for l in l_fps_names]  # remove s for smoothed images
        rmap_name = "rmap_str_smoothed_888.nii"  # 
        str_add = "str"

    conn_arr, l_per = get_performances_and_flattened_fingerprints(l_fps_names, df_ind_ch)
    rmap_arr_np = np.nan_to_num(rmap.get_RMAP(np.array(conn_arr).T, np.array(l_per)))

    rmap.save_Nii(rmap_arr_np, rmap.affine, name=os.path.join(path_out, rmap_name))


    # leave one ch. out Cross Validation
    per_left_out, per_predict = rmap.leave_one_ch_out_cv(
        l_fps_names, l_fps_dat, l_per
    )

    df_plt_corr["test_performance"] = per_left_out
    df_plt_corr[f"{str_add}_conn_predict_lo_channel"] = per_predict

    subjects = list(df_ind_ch["sub_str"].unique())

    per_left_out, per_predict = rmap.leave_one_sub_out_cv(
        l_fps_names, l_fps_dat, l_per, subjects
    )

    df_plt_corr[f"{str_add}_conn_predict_lo_subject"] = per_predict

df_plt_corr.to_csv("cv_rmap_pred_results.csv")

# func. pred

# str. ch: p=0.0004; sub: 0.0002
x_col = "test_performance"
y_col = "func_conn_predict_lo_channel"
plt.figure(figsize=(6,4), dpi=300)
rho_ch, p_ch = nm_stats.permutationTestSpearmansRho(
    df_plt_corr[x_col],
    df_plt_corr[y_col],
    False,
    "R^2",
    5000,
)

sb.regplot(x=x_col, y=y_col, data=df_plt_corr, label="Leave one channel out", color=(53/255,183/255,121/255))

y_col = "func_conn_predict_lo_subject"
rho_sub, p_sub = nm_stats.permutationTestSpearmansRho(
    df_plt_corr[x_col],
    df_plt_corr[y_col],
    False,
    "R^2",
    5000,
)
sb.regplot(x=x_col, y=y_col, data=df_plt_corr, label="Leave one subject out", color=(49/255,104/255,142/255))
plt.legend()
plt.ylabel("predicted performance RMap corr")
plt.title("Functional Connectivity\nleave one channel out: "+r"$\rho$"+f"={np.round(rho_ch, 2)} p={np.round(p_ch, 3)}"+"\n"+\
    "leave one subject out: "+r"$\rho$"+f"={np.round(rho_sub, 2)} p={np.round(p_sub, 3)}"+"\n")
plt.savefig(
        os.path.join(path_out, "RMAP_pred_func.pdf"),
        bbox_inches="tight",
    )



# str. ch: p=0.0004; sub: 0.0002

x_col = "test_performance"
y_col = "str_conn_predict_lo_channel"
plt.figure(figsize=(6,4), dpi=300)
rho_ch, p_ch = nm_stats.permutationTestSpearmansRho(
    df_plt_corr[x_col],
    df_plt_corr[y_col],
    False,
    "R^2",
    5000,
)

sb.regplot(x=x_col, y=y_col, data=df_plt_corr, label="Leave one channel out", color=(53/255,183/255,121/255))

y_col = "str_conn_predict_lo_subject"
rho_sub, p_sub = nm_stats.permutationTestSpearmansRho(
    df_plt_corr[x_col],
    df_plt_corr[y_col],
    False,
    "R^2",
    5000,
)
sb.regplot(x=x_col, y=y_col, data=df_plt_corr, label="Leave one subject out", color=(49/255,104/255,142/255))
plt.ylabel("predicted performance RMap corr")
plt.title("Structural Connectivity\nleave one channel out: "+r"$\rho$"+f"={np.round(rho_ch, 2)} p={np.round(p_ch, 3)}"+"\n"+\
    "leave one subject out: "+r"$\rho$"+f"={np.round(rho_sub, 2)} p={np.round(p_sub, 3)}"+"\n")
plt.savefig(
        os.path.join(path_out, "RMAP_pred_str_smooth_888.pdf"),
        bbox_inches="tight",
    )

