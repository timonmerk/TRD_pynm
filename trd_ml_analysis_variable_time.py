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
from imblearn import over_sampling
import trd_ml_set_decoder_3500 as trd_utils


PATH_FEATURES = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\features_computed"
PATH_RESULTS = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_comp_labels"

subjects = [
    f for f in os.listdir(PATH_FEATURES) if f.startswith("effspm8") and "KSC" not in f
]
GET_AVG_LABEL = True
pd_data = []

PLS = -2
UNPLS = -1
NTR = -3
ALL = -4

LABEL = ALL
mean_acc = []
epoch_out = {}
data_all = []
label_all = []

SW_Features_Select = trd_utils.SW_Features_Select

# integer_encoded_names: 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
roc_res = {}
cm_res = {}

FIX_LABEL_TO_3500ms = False
FIX_LABEL_TO_750_2300ms = False  # if both are false only the 1000ms
FIX_LABEL_TO_1000_2200ms = False

NTR_VS_EMOTION = True
NTR_VS_PLS = False
NTR_VS_UNPLS = False
PLS_VS_UNPLS = False
PLS_VS_UNPLS_VS_NTR = False
PLS_VS_UNPLS_VS_NTR_VS_REST = False

FFT_ONLY = False
SW_ONLY = False
FFT_SW_ONLY = True

coef_l = []

for sub in subjects:

    decoder_3500 = trd_utils.get_analyzer_3500(
        sub,
        NTR_VS_EMOTION,
        PLS_VS_UNPLS,
        PLS_VS_UNPLS_VS_NTR,
        PLS_VS_UNPLS_VS_NTR_VS_REST,
        NTR_VS_PLS,
        NTR_VS_UNPLS,
    )
    decoder_3500.set_data_ind_channels()

    analyzer = nm_analysis.Feature_Reader(
        feature_dir=PATH_FEATURES,
        feature_file=sub,
        binarize_label=False,
    )

    if FIX_LABEL_TO_3500ms is True:
        # fix labels ranging 3.5s
        label_new = np.copy(analyzer.label)
        jumps = np.diff(analyzer.label)
        idx_fix = np.where(jumps > 0)[0]
        for idx in idx_fix:
            for i in range(25):  # after 1s the label is set to "no stim" (0)
                label_new[idx + 10 + i] = jumps[idx]
        analyzer.feature_arr["ALL"] = label_new

    if FIX_LABEL_TO_750_2300ms is True:
        label_new = np.copy(analyzer.label)
        jumps = np.diff(analyzer.label)
        idx_fix = np.where(jumps > 0)[0]
        for idx in idx_fix:
            for i in np.arange(8):
                label_new[idx + i] = 0
            for i in np.arange(15):
                label_new[idx + 8 + i] = jumps[idx]

        analyzer.feature_arr["ALL"] = label_new

    if FIX_LABEL_TO_1000_2200ms is True:
        label_new = np.copy(analyzer.label)
        jumps = np.diff(analyzer.label)
        idx_fix = np.where(jumps > 0)[0]
        for idx in idx_fix:
            for i in np.arange(10):
                label_new[idx + i] = 0
            for i in np.arange(12):
                label_new[idx + 10 + i] = jumps[idx]

        analyzer.feature_arr["ALL"] = label_new

    # select here idx_select
    idx_select, analyzer = trd_utils.get_idx_select(
        analyzer,
        NTR_VS_EMOTION=NTR_VS_EMOTION,
        PLS_VS_UNPLS=PLS_VS_UNPLS,
        PLS_VS_UNPLS_VS_NTR=PLS_VS_UNPLS_VS_NTR,
        PLS_VS_UNPLS_VS_NTR_VS_REST=PLS_VS_UNPLS_VS_NTR_VS_REST,
        NTR_VS_PLS=NTR_VS_PLS,
        NTR_VS_UNPLS=NTR_VS_UNPLS,
    )

    analyzer.feature_arr = analyzer.feature_arr.loc[
        idx_select,
        :,
    ].iloc[:, :-5]

    feature_names = []
    for f in analyzer.feature_arr.columns:
        if (FFT_ONLY is True or FFT_SW_ONLY is True) and "fft" in f:
            feature_names.append(f)
        if SW_ONLY is True or FFT_SW_ONLY is True:
            for k in SW_Features_Select:
                if k in f:
                    feature_names.append(f)

    if len(feature_names) > 0:
        analyzer.feature_arr = analyzer.feature_arr[feature_names]

    analyzer.feature_arr.columns = list(
        map(trd_utils.fix_name_columns, analyzer.feature_arr.columns)
    )

    decoder = nm_decode.Decoder(
        features=analyzer.feature_arr,
        label=analyzer.label,
        label_name=analyzer.label_name,
        used_chs=analyzer.used_chs,
        sfreq=analyzer.sfreq,
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        save_coef=True,
        model=linear_model.LogisticRegression(
            multi_class="multinomial", class_weight="balanced"
        ),
        eval_method=metrics.balanced_accuracy_score,
        cv_method="NonShuffledTrainTestSplit",  # model_selection.KFold(
        # n_splits=3, random_state=None, shuffle=False
        # ), #
        get_movement_detection_rate=False,
        mov_detection_threshold=0.5,
        min_consequent_count=3,
        threshold_score=False,
        bay_opt_param_space=None,
        STACK_FEATURES_N_SAMPLES=False,
        time_stack_n_samples=5,
        use_nested_cv=False,
        VERBOSE=False,
        undersampling=False,
        oversampling=True,
        mrmr_select=False,
        cca=False,
        pca=False,
        model_save=True,
    )

    analyzer.decoder = decoder

    performances = analyzer.run_ML_model(
        estimate_channels=True, estimate_all_channels_combined=False
    )

    df = analyzer.get_dataframe_performances(performances)

    # if sub == "effspm8_SCH_EMO":
    df_q = df.query('ch_type == "electrode ch"')
    best_ch = df_q.iloc[df_q["performance_test"].argmax()]["ch"]
    coef = performances[""][best_ch]["coef"]  # best channels
    featurenames = [f[8:] for f in analyzer.feature_arr.columns if best_ch in f]
    coef_l.append(coef)

    df["sub"] = sub
    pd_data.append(df)

    df_q = df.query('ch_type == "electrode ch"')
    best_ch = df_q.iloc[df_q["performance_test"].argmax()]["ch"]

    # for ch in performances[""].keys():
    model_train = performances[""][best_ch]["model_save"][0]
    y_pr_proba = model_train.predict_proba(decoder_3500.ch_ind_data[best_ch])

    ch_epochs = []
    for label_ in [0, 1]:
        y_te = decoder_3500.label
        y_pr = np.rint(y_pr_proba[:, label_])
        if label_ == 0:
            y_te = 1 - y_te
        y_te_pr_epochs, y_te_epochs = analyzer.get_epochs(
            np.expand_dims(y_pr, axis=(1, 2)),
            y_te,
            epoch_len=7,
            sfreq=10,
            threshold=0.1,
        )
        y_te_pr_epochs = np.squeeze(y_te_pr_epochs)
        ch_epochs.append(y_te_pr_epochs)
    mean_acc.append(np.concatenate(ch_epochs).mean(axis=0)[35:])


# plt coefficients:

coef = np.abs(np.squeeze(np.array(coef_l))).sum(axis=0)

plt.figure(figsize=(6, 3), dpi=300)
idx = np.argsort(coef)[::-1]
plt.bar(np.arange(coef.shape[0]), coef[idx], color="black")
plt.xticks(np.arange(coef.shape[0]), np.array(featurenames)[idx])
plt.xticks(rotation=90)
plt.ylabel("Feature Importance")
plt.title("Examplary Feature Importance Best channel")
plt.savefig(
    r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\Figures_17_08\feature_importance_sum.pdf",
    bbox_inches="tight",
)
# get best channel bar
df = pd.concat(pd_data)

PATH_RESULTS = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\Figures_17_08"
# plot performances here as well


def get_df_acc(mean_acc, time_length: int = 35, time_start: int = 100):
    df_acc = pd.DataFrame()

    arr_acc_pls = np.stack([f for f in mean_acc])[:, :time_length]  # PLS
    for i in range(arr_acc_pls.shape[0]):
        for t in range(time_length):
            df_acc = df_acc.append(
                {
                    "sub": i,
                    "Accuracy": arr_acc_pls[i, t],
                    "Time [ms]": np.round(time_start + t * 100),
                },
                ignore_index=True,
            )

    return df_acc


def time_plot_simple(
    time_length: int = 35,
    time_start: int = 100,
    PATH_RESULTS: str = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\700_2200",
):

    df_acc = get_df_acc(mean_acc, time_length, time_start)

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
        label="mean accuracy",
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
        np.arange(time_length),
        np.arange(time_start, time_start + time_length * 100, 100),
        rotation=90,
    )
    plt.title(f"time range: {os.path.basename(PATH_RESULTS)} ms")

    plt.savefig(
        PATH_RESULTS,
        bbox_inches="tight",
    )


df_acc = get_df_acc(
    mean_acc, time_length=35, time_start=100
)  #  we always want to plot the whole range


time_plot_simple(
    time_length=35,
    time_start=100,
    PATH_RESULTS=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\Figures_17_08\time_plt_unpls_vs_ntrl.pdf",
)


acc = df.groupby("sub")["performance_test"].max()


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

# map them UPDRS and UPDRS improvement
df_plt = pd.DataFrame()
df_plt["sub"] = ["JUN", "KOR", "MIC", "NIL", "OHL", "SCH", "THI", "WES"]
df_plt["acc"] = np.array(acc)

df_plt["bdi"] = df_plt["sub"].map(bdi_mapping)
df_plt["bdi_change"] = df_plt["sub"].map(bdi_change_mapping)

x_col = "acc"
y_col = "bdi_change"
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


plt.savefig(
    os.path.join(PATH_RESULTS, "corr_bdi_change_dbs_0_1000_max.pdf"),
    bbox_inches="tight",
)

y_col = "bdi"
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


plt.savefig(
    os.path.join(PATH_RESULTS, "corr_bdi_dbs_0_1000_max.pdf"),
    bbox_inches="tight",
)

col_ = "Time [ms]"

idx = df_acc.groupby("sub")["Accuracy"].transform(np.mean) == df_acc["Accuracy"]
df_best = df_acc[idx].groupby("sub").mean()
df_best.mean() == 0.73 + -0.04  # performances
df_best.std() == 839.6 + -304  # time point
acc = df_best["Accuracy"]

acc = df_acc.groupby("Time [ms]")["Accuracy"].mean()

col_ = "Time [ms]"
val_ = 600.0  # 500.00 funktioniert ganz gut
acc = df_acc[df_acc[col_] == val_]["Accuracy"]

# get mean time point of max

# map them UPDRS and UPDRS improvement


###

nm_plots.plot_df_subjects(
    df=df,
    y_col="performance_test",
    x_col="sub",
    hue=None,
    title="All channel performances 700-2200 ms",
    PATH_SAVE=os.path.join(
        PATH_RESULTS,
        "per_all_ch_700_2200.pdf",
    ),
)


p_vals = []
for val_ in np.arange(700, 2200, 100):  # 100, 1100, 100
    _, p = nm_stats.permutationTest(
        np.array(df_acc[df_acc[col_] == val_]["Accuracy"]),
        np.array([0.5 for _ in range(8)]),
        plot_=False,
        x_unit="ba",
    )
    p_vals.append(p)
# 0->1000
# p_vals = [0.4084, 0.001, 0.0014, 0.0004, 0.0002, 0.0, 0.0002, 0.0, 0.0, 0.0]

acc = np.array(df_acc[df_acc[col_] != 100.0].groupby("sub")["Accuracy"].mean())

acc = np.array(
    df_acc[df_acc[col_] != 700.0 and df_acc[col_] != 800.0]
    .groupby("sub")["Accuracy"]
    .mean()
)

acc = df.groupby("sub")["performance_test"].max()

# for 700 - 2200 select the maximum

# peak ist bei 600 ms

# select the vals subject wise
col_ = "Time [ms]"
val_ = 600.0
acc = df_acc[df_acc[col_] == val_]["Accuracy"]

# select performance based on trained range [700-2200]
acc = df.groupby("sub")["performance_test"].mean()
