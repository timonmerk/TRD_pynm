import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sb

from py_neuromodulation import nm_RMAP, nm_plots, nm_stats, nm_IO

PATH_RES = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics"
per_predict_lo_ch = nm_IO.loadmat(os.path.join(PATH_RES, "fiberfilt_performances_lochout_cv.mat"))["Ihat"]
per_predict_lo_subject = nm_IO.loadmat(os.path.join(PATH_RES, "fiberfilt_performances_losubout_cv.mat"))["Ihat"]
per_predict_true = nm_IO.loadmat(os.path.join(PATH_RES, "fiberfilt_true_performances.mat"))["I"]

df_plt = pd.DataFrame()
df_plt["test_performance"] = per_predict_true
df_plt["fiberfilt_predict_lo_channel"] = per_predict_lo_ch
df_plt["fiberfilt_predict_lo_subject"] = per_predict_lo_subject

x_col = "test_performance"
y_col = "fiberfilt_predict_lo_channel"
plt.figure(figsize=(6,4), dpi=300)
rho_ch, p_ch = nm_stats.permutationTestSpearmansRho(
    df_plt[x_col],
    df_plt[y_col],
    False,
    "R^2",
    5000,
)

sb.regplot(x=x_col, y=y_col, data=df_plt, label="Leave one channel out", color=(53/255,183/255,121/255))

y_col = "fiberfilt_predict_lo_subject"
rho_sub, p_sub = nm_stats.permutationTestSpearmansRho(
    df_plt[x_col],
    df_plt[y_col],
    False,
    "R^2",
    5000,
)
plt.legend()
sb.regplot(x=x_col, y=y_col, data=df_plt, label="Leave one subject out", color=(49/255,104/255,142/255))
plt.ylabel("Predicted performance T value")
plt.title("Fiber Filtering\nleave one channel out: "+r"$\rho$"+f"={np.round(rho_ch, 2)} p={np.round(p_ch, 3)}"+"\n"+\
    "leave one subject out: "+r"$\rho$"+f"={np.round(rho_sub, 2)} p={np.round(p_sub, 3)}"+"\n")
plt.savefig(
        os.path.join(PATH_RES, "FiberFiltPredictPerformances.pdf"),
        bbox_inches="tight",
    )
