import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os

df_elec = pd.read_csv(
    os.path.join(
        "scripts", "Analysis connectomics", "electrodes_with_stim_contacts_bipolar.csv"
    )
)

# iterate through every subject and through sides L and R, and check if the channel with highest per is also the one with the stim contact
l_sub = df_elec["sub"].unique()
l_sides = ["L", "R"]
l_bool = []
dict_out = {}
for sub in l_sub:
    for side in l_sides:
        df_sub = df_elec.query("sub == @sub and Side == @side")
        ch_max = df_sub["per_new"].reset_index()["per_new"].idxmax()
        selected = ch_max in np.where(df_sub["Stim selection"] == 1)[0]
        l_bool.append(selected)
        dict_out[sub + "_" + side] = selected

np.sum(l_bool)
# in 5 / 16 hemispheres the best decoding channel is the one with highest decoding performance
# which is exactly chance..
# that's 4 / 8 subjects

# for the bipolar channel recordings: 12 out of 16 hemispheres overlapped
# all patients had hemispheres that overlapped
# Left: 6/8 Right: 6/8 patients
#

# stim contacts were selected within 6 month

from py_neuromodulation import nm_stats
import seaborn as sb

df_plt = pd.read_csv("performances_0_1000_ntr_vs_rest_final.csv")
plt.figure(figsize=(8, 4), dpi=300)
plt.subplot(1, 2, 1)
sb.boxplot(x="overlap", y="acc", data=df_plt, palette="viridis")
sb.swarmplot(x="overlap", y="acc", data=df_plt, color=".25", palette="viridis")
plt.title("Decoding performance of patients \nwith and without overlap")
plt.xlabel("Best-channel and stimulation contact overlap")
#plt.tight_layout()
#plt.figure(figsize=(3, 4), dpi=300)
plt.subplot(1, 2, 2)
sb.boxplot(x="overlap", y="bdi_change_24_months", data=df_plt)
sb.swarmplot(x="overlap", y="bdi_change_24_months", data=df_plt, color=".25")
plt.title("Symptom improvement of patients \nwith and without overlap")
plt.xlabel("Best-channel and stimulation contact overlap")

plt.suptitle("")
plt.tight_layout()

x_col = "acc"
y_col = "bdi_change_6_months"
data = df_plt
plt.figure(figsize=(3, 4), dpi=300)
rho, p = nm_stats.permutationTestSpearmansRho(
    data[x_col],
    data[y_col],
    False,
    "R^2",
    5000,
)
sb.regplot(x=x_col, y=y_col, data=data)
plt.title(f"{y_col}~{x_col} rho={np.round(rho, 2)} p={np.round(p, 2)}")
#
