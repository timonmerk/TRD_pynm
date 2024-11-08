import enum
from py_neuromodulation import nm_decode, nm_analysis, nm_plots

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


def fix_name_columns(feature_name: str):
    if feature_name.startswith("burst"):
        feature_str = "bursts"
    elif feature_name.startswith("nolds"):
        feature_str = "nolds"
    else:
        return feature_name
    str_start = feature_name.find("_") + 1
    str_end = feature_name.find("_") + 8
    ch_name = feature_name[str_start:str_end]
    feature_name_new = ch_name + "_" + feature_str + feature_name[str_end:]
    return feature_name_new


PATH_FEATURES = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\features_computed"
PATH_RESULTS = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_comp_labels"

subjects = [
    f for f in os.listdir(PATH_FEATURES) if f.startswith("effspm8") and "KSC" not in f
]
GET_AVG_LABEL = False
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

SW_Features_Select = [
    "Sharpwave_Max_prominence_range_5_80",
    "Sharpwave_Max_prominence_range_5_30",
    "Sharpwave_Mean_interval_range_5_30",
    "Sharpwave_Mean_interval_range_5_80",
    "Sharpwave_Max_sharpness_range_5_30",
    "Mean_decay_time_range_5_30",
    "Mean_rise_time_range_5_30",
]

# integer_encoded_names: 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
roc_res = {}
cm_res = {}

FIX_LABEL_TO_3500ms = False
FIX_LABEL_TO_750_2300ms = False  # if both are false only the 1000ms
FIX_LABEL_TO_1000_2200ms = False

NTR_VS_EMOTION = False
PLS_VS_UNPLS = True
PLS_VS_UNPLS_VS_NTR = False
PLS_VS_UNPLS_VS_NTR_VS_REST = False

FFT_ONLY = False
SW_ONLY = False
FFT_SW_ONLY = True

for sub in subjects:
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

    if NTR_VS_EMOTION is True:
        idx_select = analyzer.feature_arr["ALL"] != 0
        analyzer.label = np.array(
            analyzer.feature_arr.loc[
                idx_select,
                :,
            ].iloc[:, LABEL]
        )
        analyzer.label[np.where(analyzer.label == 1)] = 0  # NTR
        analyzer.label[np.where(analyzer.label != 0)] = 1  # PLS + UNPLS

    elif PLS_VS_UNPLS is True:
        idx_select = (analyzer.feature_arr["ALL"] != 0) & (
            analyzer.feature_arr["ALL"] != 1
        )
        analyzer.label = np.array(
            analyzer.feature_arr.loc[
                idx_select,
                :,
            ].iloc[:, LABEL]
        )  # keep only PLS and UNPLS labels
        analyzer.label[np.where(analyzer.label == 2)] = 0  # PLS
        analyzer.label[np.where(analyzer.label != 0)] = 1  # UNPLS

    elif PLS_VS_UNPLS_VS_NTR is True:
        idx_select = analyzer.feature_arr["ALL"] != 0
        analyzer.label = np.array(
            analyzer.feature_arr.loc[
                idx_select,
                :,
            ].iloc[:, LABEL]
        )
    elif PLS_VS_UNPLS_VS_NTR_VS_REST is True:
        idx_select = np.arange(analyzer.label.shape[0])

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
        map(fix_name_columns, analyzer.feature_arr.columns)
    )

    analyzer.set_decoder(
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        save_coef=True,
        model=linear_model.LogisticRegression(
            multi_class="multinomial", class_weight="balanced"
        ),  #
        # model=xgboost.XGBClassifier(),  # ,   # catboost.CatBoostClassifier(),
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
        use_nested_cv=True,
        VERBOSE=False,
        undersampling=False,
        oversampling=True,
        mrmr_select=False,
        cca=False,
        pca=False,
    )

    # analyzer.decoder.feature_names = list(analyzer.decoder.features.columns)

    performances = analyzer.run_ML_model(
        estimate_channels=True, estimate_all_channels_combined=False
    )

    df = analyzer.get_dataframe_performances(performances)
    df["sub"] = sub
    pd_data.append(df)

    if GET_AVG_LABEL is True:
        label_out = {}
        # get here channel based on InnerCV_performance_test

        df_q = df.query('ch_type == "electrode ch"')
        # best_ch = df_q.iloc[df_q["performance_test"].argmax()]["ch"]

        idx = df_q.groupby(["sub"])["InnerCV_performance_test"].idxmax()
        best_ch = df_q.iloc[idx]["ch"].iloc[0]

        for label_ in [0, 1]:  # 1,
            y_te = (
                np.concatenate(analyzer.decoder.ch_ind_results[best_ch]["y_test"])
                # - 2
            )

            y_te_pr = np.rint(
                np.concatenate(analyzer.decoder.ch_ind_results[best_ch]["y_test_pr"])[
                    :, label_
                ]
            )

            if label_ == 0:
                y_te = 1 - y_te

            # y_te[np.where(y_te != label_)[0]] = 0
            # y_te[np.where(y_te == label_)[0]] = 1
            y_te_pr_epochs, y_te_epochs = analyzer.get_epochs(
                np.expand_dims(y_te_pr, axis=(1, 2)),
                y_te,
                epoch_len=7,
                sfreq=10,
                threshold=0.1,
            )
            y_te_pr_epochs = np.squeeze(y_te_pr_epochs)
            acc_ = (
                np.sum(y_te_pr_epochs[:, 35:] == 1, axis=0)  # 35 + 13
                / y_te_epochs.shape[0]
            )  # 35:
            label_out[label_] = acc_
            label_out["label"] = y_te_epochs.mean(axis=0)

        mean_acc.append(label_out)

    if sub not in roc_res:
        roc_res[sub] = {}
        cm_res[sub] = {}

    try:
        for ch in analyzer.decoder.ch_ind_results.keys():
            # fpr, tpr, thr = metrics.roc_curve(
            #    analyzer.decoder.ch_ind_results[ch]["y_test"][0],
            #    analyzer.decoder.ch_ind_results[ch]["y_test_pr"][0][:, 1],
            # )

            # roc_res[sub][ch] = {}
            # roc_res[sub][ch]["fpr"] = fpr
            # roc_res[sub][ch]["tpr"] = tpr
            # roc_res[sub][ch]["thr"] = thr

            cm_res[sub][ch] = metrics.confusion_matrix(
                analyzer.decoder.ch_ind_results[ch]["y_test"][0],  # -1 for 3 class
                np.argmax(analyzer.decoder.ch_ind_results[ch]["y_test_pr"][0], axis=1),
                normalize="true",
            )
    except:
        print("no calc of ROC")


df = pd.concat(pd_data)

time_plot_simple(
    time_length=15,
    time_start=700,
    PATH_RESULTS=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\700_2200\700_2200.pdf",
)

time_plot_simple(
    time_length=10,
    time_start=100,
    PATH_RESULTS=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_1000\0_1000.pdf",
)

time_plot_simple(
    time_length=35,
    time_start=100,
    PATH_RESULTS=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_3500\0_3500.pdf",
)

# analysis for the limited time range (700-2200 ms)
# 1. plot performances
df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_best_sub_ind_ch = df_ind_ch.iloc[idx]
PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\700_2200"
nm_plots.plot_df_subjects(
    df=df_best_sub_ind_ch,
    y_col="performance_test",
    x_col="sub",
    hue=None,
    title="All channel performances 700-2200 ms",
    PATH_SAVE=os.path.join(
        PATH_OUT,
        "per_all_ch_.pdf",
    ),
)

# 0-3500
df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_best_sub_ind_ch = df_ind_ch.iloc[idx]
PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_3500"
nm_plots.plot_df_subjects(
    df=df_best_sub_ind_ch,
    y_col="performance_test",
    x_col="sub",
    hue=None,
    title="All channel performances 0-3500 ms",
    PATH_SAVE=os.path.join(
        PATH_OUT,
        "per_all_ch_0_3500.pdf",
    ),
)

# 0-1000ms
df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_best_sub_ind_ch = df_ind_ch.iloc[idx]
PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_1000"
nm_plots.plot_df_subjects(
    df=df_best_sub_ind_ch,
    y_col="performance_test",
    x_col="sub",
    hue=None,
    title="All channel performances 0-1000 ms",
    PATH_SAVE=os.path.join(
        PATH_OUT,
        "per_all_ch_0_1000.pdf",
    ),
)


# get best channel bar
df = pd.concat(pd_data)


df.to_pickle(os.path.join(PATH_RESULTS, "df_sw_fft_ntr_vs_plsunpls.p"))

# pickle also averaged accuracy, confusion matrix and roc curves
with open(os.path.join(PATH_RESULTS, "roc_res.pickle"), "wb") as handle:
    pickle.dump(roc_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(PATH_RESULTS, "cm_res.pickle"), "wb") as handle:
    pickle.dump(cm_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(PATH_RESULTS, "mean_acc.pickle"), "wb") as handle:
    pickle.dump(mean_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)


# multiclass confusion matrix:
df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_best_sub_ind_ch = df_ind_ch.iloc[idx]
cm_l = []
for idx, row in df_best_sub_ind_ch.iterrows():
    cm_l.append(cm_res[row["sub"]][row["ch"]])

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=np.array(cm_l).mean(axis=0),
    display_labels=["Rest", "Neutral", "Pleasant", "Unpleasant"],  #
)
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
disp.plot(ax=ax)
disp.ax_.set_title("Mean Best channel confusion matrix")
# cbar = plt.colorbar()
# cbar.set_label("Accuracy")
plt.savefig(
    "cm_4class_neu.pdf",
    bbox_inches="tight",
)
# correlate the BDI scores to the misclassified unpls -> ntr scores
miss_unpl_ntr = np.array(cm_l)[:, 3, 3]
bdi_scores = [33, 46, 35, 41, 22, 43, 36, 57]
df_bdi_cm = pd.DataFrame()
df_bdi_cm["miss_unpl"] = miss_unpl_ntr
df_bdi_cm["bdi"] = bdi_scores
nm_plots.reg_plot(x_col="miss_unpl", y_col="bdi", data=df_bdi_cm, out_path_save=None)


# save optionally the output dataframe according to used features, e.g. df.to_pickle("df_fft.p")
df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
df_best_sub_ind_ch = df_ind_ch.iloc[idx]


def plot_best_ch_counts(df_best_sub_ind_ch: pd.DataFrame):
    # how many times which channel performed best?

    df_best_sub_ind_ch.ch.value_counts().sort_values().plot(kind="barh")

    plt.xlabel("Number of times best")
    plt.ylabel("Channel name")


def plot_roc_curves(df_best_sub_ind_ch: pd.DataFrame, roc_res: dict):
    plt.figure(dpi=300)
    lw = 2

    for idx, row in df_best_sub_ind_ch.iterrows():
        plt.plot(
            roc_res[row["sub"]][row["ch"]]["fpr"],
            roc_res[row["sub"]][row["ch"]]["tpr"],
            label=row["sub"],
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def plot_cm_best_best_ch(df_best_sub_ind_ch: pd.DataFrame, cm_res: dict):
    # Mean Confusion Matrix
    cm_l = []
    for idx, row in df_best_sub_ind_ch.iterrows():
        cm_l.append(cm_res[row["sub"]][row["ch"]])

    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm_l).mean(axis=0),
        display_labels=["NTR", "PLS+UNPLS"],
    )
    disp.plot()
    disp.ax_.set_title("Mean Best channel confusion matrix")

    # plt.figure(dpi=300)
    # plt.imshow(np.array(cm_l).mean(axis=0), aspect="auto")
    # plt.xticks([0, 1], ["PLS", "UNPLS"])
    # plt.yticks([0, 1], ["PLS", "UNPLS"])
    # plt.show()


def plot_performance_best_ch(df):
    nm_plots.plot_df_subjects(
        df=df,
        y_col="performance_test",
        x_col="sub",
        hue="ch_type",
        title="best channel performances",
        PATH_SAVE=os.path.join(
            r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion",
            "per_sub_3500.pdf",
        ),
    )


def boxplot_coef(feature_names: list):
    # make a boxplot with coefficients

    df_ = pd.read_pickle("df_fft_sw.p")  # df_fft_sw df_Sharpwave
    feature_names_bp = [f[8:] for f in feature_names if f.startswith("Cg25R01")]

    df_ind_ch = df_.query("ch_type == 'electrode ch'")

    idx = df_ind_ch.groupby(["sub"])["performance_test"].idxmax()

    df_mean_sub_ind_ch = df_.iloc[idx]

    df_plt = pd.DataFrame()
    for _, row in df_mean_sub_ind_ch.iterrows():
        for f_idx, f in enumerate(feature_names_bp):
            df_plt = df_plt.append(
                {
                    "sub": row["sub"],
                    "coef": row["coef"][0, f_idx],
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
        PATH_SAVE=r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\results_paper_figures_2\LM_fft_coef_sub.png",
    )


def get_df_acc(mean_acc: dict, time_length: int, time_start: int):
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
    return df_acc


def time_plot_simple(
    time_length: int = 10,
    time_start: int = 100,
    PATH_RESULTS: str = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_1000",
):

    df_acc = get_df_acc(mean_acc, time_length, time_start)

    alpha_box = 0.4
    plt.figure(figsize=(5, 3), dpi=300)

    df_resp = df_acc.groupby(["Stimulus", "Time [ms]"]).median().reset_index()
    plt.plot(
        np.arange(time_length),
        df_resp.query("Stimulus == 'neutral'")["Accuracy"],
        label="Neutral",
        color="green",
    )

    plt.plot(
        np.arange(time_length),
        df_resp.query("Stimulus == 'pls+unpls'")["Accuracy"],
        label="Neutral",
        color="blue",
    )

    df_mean_median = (
        df_acc.groupby(["Stimulus", "Time [ms]"])
        .median()
        .reset_index()
        .groupby("Time [ms]")
        .mean()
        .reset_index()
    )

    plt.plot(
        np.arange(time_length),
        df_mean_median["Accuracy"],
        linewidth=4,
        alpha=0.7,
        label="mean",
        color="black",
    )

    plt.plot(
        np.arange(time_length),
        [0.5 for i in range(time_length)],
        label="chance",
        color="gray",
    )

    plt.legend()

    plt.ylabel("Balanced Accuracy")
    plt.xlabel("Time [ms]")
    plt.xticks(
        np.arange(time_length),
        np.arange(time_start, time_start + time_length * 100, 100),
        rotation=90,
    )
    plt.title(f"time range: {os.path.basename(PATH_RESULTS)[:-4]} ms")

    plt.savefig(
        PATH_RESULTS,
        bbox_inches="tight",
    )


def time_plot(
    mean_acc,
    time_length: int = 10,
    time_start: int = 100,
    PATH_RESULTS: str = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\results_new_time_range\0_1000",
):

    df_acc = get_df_acc(mean_acc, time_length, time_start)
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
        data=df_acc,
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
        data=df_acc,
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


# bdi correlation plot


def bdi_correlation_plot(df):
    # from file: "C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\excel sheets BDI scores"
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

    nm_plots.reg_plot(
        x_col="performance_test", y_col="bdi_change", data=df_mean_sub_ind_ch
    )
