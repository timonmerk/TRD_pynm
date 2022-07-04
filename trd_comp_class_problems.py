import enum
from py_neuromodulation import nm_decode, nm_analysis, nm_plots

import pandas as pd
import numpy as np
import os
import _pickle as cPickle
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


# 1. read all df's and then select only 

df_names = ["df_sw_fft_pls_vs_unpls.p",
    "df_sw_fft_ntr_vs_plsunpls.p",
    "df_sw_fft_ntr_vs_pls_vs_unpls.p",
    "df_sw_fft_ntr_vs_pls_vs_unpls_vs_rest.p",
]
df_list = []

for df_name in df_names:

    df = pd.read_pickle(os.path.join(PATH_RESULTS, df_name))
    df["class_problem"] = df_name[10:-2]

    df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
    idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
    df_best_sub_ind_ch = df_ind_ch.iloc[idx]
    df_list.append(df_best_sub_ind_ch)
    df_list.append(df.query("ch_type == 'all ch combinded'"))

df_comb = pd.concat(df_list)

nm_plots.plot_df_subjects(
        df=df_comb,
        y_col="performance_test",
        x_col="class_problem",
        hue="ch_type",
        title="best channel performances",
        PATH_SAVE=os.path.join(PATH_RESULTS, "LM_per_different_class_problems.png"),
    )
