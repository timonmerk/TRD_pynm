import enum
from py_neuromodulation import nm_decode, nm_analysis, nm_plots, nm_stats

import pandas as pd
import numpy as np
import os
import pickle
import xgboost
import catboost
import seaborn as sb
from sklearn import (
    metrics,
    linear_model,
    model_selection,
    ensemble,
    preprocessing,
)
from matplotlib import pyplot as plt
from imblearn import over_sampling

PATH_RESULTS = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_comp_labels"
PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\update_Figures_11_07_22"


def plot_performance_feature_comparison():

    df_sw = pd.read_pickle(os.path.join(PATH_RESULTS, "df_sw_ntr_vs_plsunpls.p"))
    df_fft = pd.read_pickle(os.path.join(PATH_RESULTS, "df_fft_ntr_vs_plsunpls.p"))
    df_swfft = pd.read_pickle(os.path.join(PATH_RESULTS, "df_sw_fft_ntr_vs_plsunpls.p"))
    df_sw["features_used"] = "Sharpwave"
    df_fft["features_used"] = "fft"
    df_swfft["features_used"] = "Sharpwave_fft"

    df_ind_ch = df_sw.query("ch_type == 'electrode ch'").reset_index()
    idx = df_ind_ch.groupby(["sub"])["performance_test"].idxmax()
    df_sw_best = df_ind_ch.iloc[idx]

    df_ind_ch = df_fft.query("ch_type == 'electrode ch'").reset_index()
    idx = df_ind_ch.groupby(["sub"])["performance_test"].idxmax()
    df_fft_best = df_ind_ch.iloc[idx]

    df_ind_ch = df_swfft.query("ch_type == 'electrode ch'").reset_index()
    idx = df_ind_ch.groupby(["sub"])["performance_test"].idxmax()
    df_swfft_best = df_ind_ch.iloc[idx]

    df_comb = pd.concat([df_sw, df_fft, df_swfft])
    df_comb_ind = pd.concat([df_sw_best, df_fft_best, df_swfft_best])

    df_plt = pd.concat([df_comb.query("ch_type == 'all ch combinded'"), df_comb_ind])

    df = df_comb_ind  # df_plt
    df = df.reset_index()
    y_col = "performance_test"
    x_col = "ch_type"
    x_col = "features_used"
    title = ("best channel performances",)
    PATH_SAVE = os.path.join(PATH_RESULTS, "LM_comp_per_best_ind_diff.png")

    alpha_box = 0.4
    plt.figure(figsize=(5, 3), dpi=300)
    sb.boxplot(
        x=x_col,
        y=y_col,
        # hue=hue,
        data=df,
        palette="viridis",
        showmeans=False,
        boxprops=dict(alpha=alpha_box),
        showcaps=True,
        showbox=True,
        showfliers=False,
        notch=False,
        whiskerprops={"linewidth": 2, "zorder": 10, "alpha": alpha_box},
        capprops={"alpha": alpha_box},
        medianprops=dict(linestyle="-", linewidth=5, color="gray", alpha=alpha_box),
    )

    ax = sb.stripplot(
        x=x_col,
        y=y_col,
        # hue=hue,
        data=df,
        palette="viridis",
        jitter=False,
        s=5,
    )

    arr_plt = np.array(
        [
            df.query("features_used == 'Sharpwave'")["performance_test"],
            df.query("features_used == 'fft'")["performance_test"],
            df.query("features_used == 'Sharpwave_fft'")["performance_test"],
        ]
    )

    for i in range(arr_plt.shape[1]):
        plt.plot(arr_plt[:, i], color="gray", linewidth=1, alpha=0.2)

    plt.ylim(
        0.5,
    )

    plt.savefig(
        os.path.join(PATH_RESULTS, "comp_fft_sw_features.pdf"),
        bbox_inches="tight",
    )


# 1. load dataframe for plotting
df = pd.read_pickle(os.path.join(PATH_RESULTS, "df_sw_fft_ntr_vs_plsunpls.p"))
df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_best_sub_ind_ch = df_ind_ch.iloc[idx]

with open(os.path.join(PATH_RESULTS, "roc_res.pickle"), "rb") as handle:
    roc_res = pickle.load(handle)
with open(os.path.join(PATH_RESULTS, "cm_res.pickle"), "rb") as handle:
    cm_res = pickle.load(handle)
with open(os.path.join(PATH_RESULTS, "mean_acc.pickle"), "rb") as handle:
    mean_acc = pickle.load(handle)


bdi_mapping = {
    "JUN": 33,
    "KOR": 46,
    "MIC": 35,
    "NIL": 41,
    "OHL": 22,
    "SCH": 43,
    "THI": 36,
    "WES": 57,
}

bdi_change_mapping = {
    "JUN": 33 - 35,
    "KOR": 46 - 36,
    "MIC": 33 - 32,
    "NIL": 41 - 37,
    "OHL": 22 - 1,
    "SCH": 43 - 31,
    "THI": 36 - 12,
    "WES": 57 - 52,
}

# df = pd.read_pickle("df_fft_sw.p")

df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
df_ind_ch["sub_str"] = df_ind_ch["sub"].str.split("_").apply(lambda x: x[1])
df_ind_ch["bdi"] = df_ind_ch.sub_str.map(bdi_mapping)
df_ind_ch["bdi_change"] = df_ind_ch.sub_str.map(bdi_change_mapping)
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_mean_sub_ind_ch = df_ind_ch.iloc[idx]

x_col = "performance_test"
y_col = "bdi_change"
data = df_mean_sub_ind_ch
plt.figure(figsize=(3, 4), dpi=300)
rho, p = nm_stats.permutationTestSpearmansRho(
    data[x_col],
    data[y_col],
    False,
    "R^2",
    5000,
)
sb.regplot(x=x_col, y=y_col, data=data)
plt.title(f"{y_col}~{x_col} p={np.round(p, 2)} rho={np.round(rho, 2)}")


plt.savefig(
    os.path.join(PATH_OUT, "corr_bdi_change_dbs.pdf"),
    bbox_inches="tight",
)

# plt absolute values of coefficients
def plt_coef():
    feature_names = [
        "fft_theta",
        "fft_alpha",
        "fft_low beta",
        "fft_high beta",
        "fft_low gamma",
        "fft_high gamma",
        "fft_HFA",
        "Sharpwave_Max_prominence_range_5_80",
        "Sharpwave_Mean_interval_range_5_80",
        "Sharpwave_Max_prominence_range_5_30",
        "Sharpwave_Mean_interval_range_5_30",
        "Sharpwave_Mean_decay_time_range_5_30",
        "Sharpwave_Mean_rise_time_range_5_30",
        "Sharpwave_Max_sharpness_range_5_30",
    ]

    feature_names_plt = [
        "fft_theta",
        "fft_alpha",
        "fft_low beta",
        "fft_high beta",
        "fft_low gamma",
        "fft_high gamma",
        "fft_HFA",
        "sw_prominence_5_80_Hz",
        "sw_interval_5_80_Hz",
        "sw_prominence_5_30_Hz",
        "sw_interval_5_30_Hz",
        "sw_decay_time_5_30_Hz",
        "sw_rise_time_5_30_Hz",
        "sw_sharpness_5_30_Hz",
    ]

    df_plt = pd.DataFrame()
    for _, row in df_best_sub_ind_ch.iterrows():
        for f_idx, f in enumerate(feature_names_plt):
            df_plt = df_plt.append(
                {
                    "sub": row["sub"],
                    "coef": np.abs(row["coef"][0, f_idx]),
                    "feature_name": f,
                },
                ignore_index=True,
            )

    nm_plots.plot_df_subjects(
        df=df_plt,
        y_col="coef",
        x_col="feature_name",
        title="Linear Model Coefficients",
        hue=None,
        PATH_SAVE=None,
    )
    plt.savefig(
        os.path.join(PATH_RESULTS, "abs_coef.pdf"),
        bbox_inches="tight",
    )


# plt subject individual performances
def plt_performances_sub():
    nm_plots.plot_df_subjects(
        df=df_plt,
        y_col="performance_test",
        x_col="sub",
        hue=None,  # "ch_type"
        title="best channel performances",
        PATH_SAVE=None,
    )
    plt.ylabel("Balanced Accuracy")
    plt.xticks(np.arange(8), [f"sub-{i+1}" for i in range(8)])
    plt.savefig(
        os.path.join(PATH_RESULTS, "sub_ind_per.pdf"),
        bbox_inches="tight",
    )


# mean confusion matrix
def plt_mean_confusion_matrix():
    cm_l = []
    for idx, row in df_best_sub_ind_ch.iterrows():
        cm_l.append(cm_res[row["sub"]][row["ch"]])

    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm_l).mean(axis=0),
        display_labels=["NTR", "PLS+UNPLS"],
    )
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    disp.plot(ax=ax)
    disp.ax_.set_title("Mean Best channel confusion matrix")
    # cbar = plt.colorbar()
    # cbar.set_label("Accuracy")
    plt.savefig(
        os.path.join(PATH_RESULTS, "cm.pdf"),
        bbox_inches="tight",
    )


# plt 2: roc curves
def plot_roc():
    plt.figure(dpi=300)
    lw = 2
    idx_sub = 0
    for idx, row in df_best_sub_ind_ch.iterrows():
        idx_sub += 1
        plt.plot(
            roc_res[row["sub"]][row["ch"]]["fpr"],
            roc_res[row["sub"]][row["ch"]]["tpr"],
            label=f"sub-{idx_sub}",
        )
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--", label="chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves best channels")
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(PATH_RESULTS, "roc_curves.pdf"),
        bbox_inches="tight",
    )

    plt.show()


# plt 1: average accuracy over time time points

# plot accuracies
def time_plot(
    time_length: int = 10,
    time_start: int = 100,
    PATH_RESULTS: str = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_1000",
):
    df_acc = pd.DataFrame()

    arr_acc_pls = np.stack([f[0] for f in mean_acc])[:, :time_length]  # PLS
    arr_acc_unpls = np.stack([f[1] for f in mean_acc])[:, :time_length]  # UNPLS
    for i in range(arr_acc_pls.shape[0]):
        for t in range(time_length):
            df_acc = df_acc.append(
                {
                    "sub": i,
                    "Accuracy": arr_acc_pls[i, t],
                    "Stimulus": "neutral",
                    "Time [ms]": np.round(time_start + t * 100),
                },
                ignore_index=True,
            )

            df_acc = df_acc.append(
                {
                    "sub": i,
                    "Accuracy": arr_acc_unpls[i, t],
                    "Stimulus": "pls+unpls",
                    "Time [ms]": np.round(time_start + t * 100),
                },
                ignore_index=True,
            )

    df = df_acc
    x_col = "Time [ms]"
    y_col = "Accuracy"
    title = "Averaged Performances Best Channels"
    hue = "Stimulus"

    alpha_box = 0.4
    plt.figure(figsize=(5, 3), dpi=300)
    sb.boxplot(
        x=x_col,
        y=y_col,
        hue=hue,
        data=df,
        palette="viridis",
        showmeans=False,
        boxprops=dict(alpha=alpha_box),
        showcaps=True,
        showbox=True,
        showfliers=False,
        notch=False,
        whiskerprops={"linewidth": 2, "zorder": 10, "alpha": alpha_box},
        capprops={"alpha": alpha_box},
        medianprops=dict(linestyle="-", linewidth=5, color="gray", alpha=alpha_box),
    )

    sb.stripplot(
        x=x_col,
        y=y_col,
        hue=hue,
        data=df,
        palette="viridis",
        dodge=True,
        alpha=alpha_box,
        s=5,
    )

    plt.title(title)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)

    # plt.legend([],[], frameon=False)
    ax = sb.pointplot(
        x="Time [ms]",
        y="Accuracy",
        data=df_acc.groupby(["Stimulus", "Time [ms]"]).median().reset_index(),
        hue="Stimulus",
        palette="viridis",
        errwidth=0,
        capsize=0,
        markers=[False, False],
        dodge=0.4,
        linewidth=10,
    )

    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(
        handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )

    plt.setp(
        ax.collections[22:], sizes=[0]
    )  # remove median horizontal line from pointplot
    plt.ylabel("Balanced Accuracy")
    plt.savefig(
        os.path.join(PATH_RESULTS, "timeplot.pdf"),
        bbox_inches="tight",
    )
