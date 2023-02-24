import enum
from re import A
from py_neuromodulation import nm_decode, nm_analysis, nm_plots, nm_stats

import pandas as pd
import numpy as np
import os
import _pickle as cPickle
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

PATH_RESULTS = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\Figures_17_08"

df_pls = pd.read_csv("df_per_max_pls_vs_ntr.csv")
df_pls["mode"] = "ntr vs pls"
df_pls_unpls = pd.read_csv("df_per_max_unpls_vs_ntr.csv")
df_pls_unpls["mode"] = "ntr vs pls/unpls"
df_unpls = pd.read_csv("df_per_max_plsunpls_vs_ntr.csv")
df_unpls["mode"] = "ntr vs unpls"

df_comb = pd.concat(
    [df_pls.reset_index(), df_pls_unpls.reset_index(), df_unpls.reset_index()]
)

# Paper performance report:
print(df_comb.query("mode == 'ntr vs pls/unpls'")["performance_test"].mean())
print(df_comb.query("mode == 'ntr vs pls/unpls'")["performance_test"].std())

print(df_comb.query("mode == 'ntr vs pls'")["performance_test"].mean())
print(df_comb.query("mode == 'ntr vs pls'")["performance_test"].std())

print(df_comb.query("mode == 'ntr vs unpls'")["performance_test"].mean())
print(df_comb.query("mode == 'ntr vs unpls'")["performance_test"].std())

nm_plots.plot_df_subjects(
    df=df_comb,
    y_col="performance_test",
    x_col="mode",
    hue=None,
    title="Best channel performances",
    PATH_SAVE=os.path.join(
        PATH_RESULTS,
        "best_ch_comp.pdf",
    ),
)


PATH_RESULTS = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\Figures_17_08\time_plt_comb.pdf"

# time plot

time_length = 35
time_start = 100

df_pls = pd.read_csv("df_time_range_0_1000_ntr_vs_pls.csv")
df_unpls = pd.read_csv("df_time_range_0_1000_ntr_vs_unpls.csv")
df_acc = pd.read_csv("df_time_range_0_1000_ntr_vs_plsunpls.csv")

plt.figure(figsize=(5, 3), dpi=300)

for sub in df_acc["sub"].unique():
    df_sub = df_acc.query("sub == @sub")
    plt.plot(
        np.arange(time_length),
        df_sub["Accuracy"],
        color="gray",
        alpha=0.2,
        linewidth=0.5,
    )
plt.plot(
    np.arange(time_length),
    df_acc.groupby("Time [ms]").mean()["Accuracy"],
    color="black",
    label="neutral vs pleasant/unpleasant",
    alpha=1,
    linewidth=2,
)

plt.plot(
    np.arange(time_length),
    df_unpls.groupby("Time [ms]").mean()["Accuracy"],
    color=(49 / 255, 104 / 255, 142 / 255),
    label="neutral vs unpleasant",
    alpha=1,
    linewidth=2,
)

plt.plot(
    np.arange(time_length),
    df_pls.groupby("Time [ms]").mean()["Accuracy"],
    color=(53 / 255, 183 / 255, 121 / 255),
    label="neutral vs pleasant",
    alpha=1,
    linewidth=2,
)


plt.plot(
    np.arange(time_length),
    [0.5 for i in range(time_length)],
    label="chance",
    color="gray",
    linestyle="dashed",
)

plt.legend()

plt.ylabel("Balanced Accuracy")
plt.xlabel("Time [ms]")
plt.xticks(
    np.arange(time_length)[::2],
    np.arange(time_start, time_start + time_length * 100, 200),
    rotation=90,
)
plt.title(f"time range: [0 1000] ms")

plt.savefig(
    PATH_RESULTS,
    bbox_inches="tight",
)
