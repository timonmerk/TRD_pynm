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
    f
    for f in os.listdir(PATH_FEATURES)
    if f.startswith("effspm8") and "KSC" not in f
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

NTR_VS_EMOTION = True
PLS_VS_UNPLS = False
PLS_VS_UNPLS_VS_NTR = False
PLS_VS_UNPLS_VS_NTR_VS_REST = False

for sub in subjects:
    analyzer = nm_analysis.Feature_Reader(
        feature_dir=PATH_FEATURES,
        feature_file=sub,
        binarize_label=False,
    )

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
        idx_select = (analyzer.feature_arr["ALL"] != 0)\
                & (analyzer.feature_arr["ALL"] != 1)
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
        if "fft" in f:
            feature_names.append(f)
        else:
            for k in SW_Features_Select:
                if k in f:
                    feature_names.append(f)

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
        cv_method= model_selection.KFold(
        n_splits=3, random_state=None, shuffle=False
        ), # "NonShuffledTrainTestSplit"
        get_movement_detection_rate=True,
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

    analyzer.decoder.feature_names = list(analyzer.decoder.features.columns)

    performances = analyzer.run_ML_model(
        estimate_channels=True, estimate_all_channels_combined=True
    )

    df = analyzer.get_dataframe_performances(performances)
    df["sub"] = sub
    pd_data.append(df)

    if GET_AVG_LABEL is True:
        label_out = {}
        df_q = df.query('ch_type == "electrode ch"')
        best_ch = df_q.iloc[df_q["performance_test"].argmax()]["ch"]
        for label_ in [0, 1]:  # 1,
            y_te = (
                np.concatenate(
                    analyzer.decoder.ch_ind_results[best_ch]["y_test"]
                )
                #- 2
            )

            y_te_pr = np.rint(
                np.concatenate(
                    analyzer.decoder.ch_ind_results[best_ch]["y_test_pr"]
                )[:, label_]
            )

            if label_ == 0:
                y_te = 1 - y_te

            # y_te[np.where(y_te != label_)[0]] = 0
            # y_te[np.where(y_te == label_)[0]] = 1
            y_te_pr_epochs, y_te_epochs = analyzer.get_epochs(
                np.expand_dims(y_te_pr, axis=(1, 2)),
                y_te,
                epoch_len=6,
                sfreq=10,
                threshold=0.1,
            )
            y_te_pr_epochs = np.squeeze(y_te_pr_epochs)
            acc_ = (
                np.sum(y_te_pr_epochs[:, 30:] == 1, axis=0)
                / y_te_epochs.shape[0]
            )
            label_out[label_] = acc_
            label_out["label"] = y_te_epochs.mean(axis=0)

        mean_acc.append(label_out)

    if sub not in roc_res:
        roc_res[sub] = {}
        cm_res[sub] = {}

    try:
        for ch in analyzer.decoder.ch_ind_results.keys():
            fpr, tpr, thr = metrics.roc_curve(
                analyzer.decoder.ch_ind_results[ch]["y_test"][0],
                analyzer.decoder.ch_ind_results[ch]["y_test_pr"][0][:, 1],
            )

            roc_res[sub][ch] = {}
            roc_res[sub][ch]["fpr"] = fpr
            roc_res[sub][ch]["tpr"] = tpr
            roc_res[sub][ch]["thr"] = thr

            cm_res[sub][ch] = metrics.confusion_matrix(
                analyzer.decoder.ch_ind_results[ch]["y_test"][0],
                np.rint(analyzer.decoder.ch_ind_results[ch]["y_test_pr"][0][:, 1]),
                normalize="true",
            )
    except:
        print("no calc of ROC")

# get best channel bar
df = pd.concat(pd_data)
df.to_pickle(os.path.join(PATH_RESULTS, "df_sw_fft_ntr_vs_plsunpls.p"))

# pickle also averaged accuracy, confusion matrix and roc curves
with open(os.path.join(PATH_RESULTS, 'roc_res.pickle'), 'wb') as handle:
    pickle.dump(roc_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
with open(os.path.join(PATH_RESULTS, 'cm_res.pickle'), 'wb') as handle:
    pickle.dump(cm_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(PATH_RESULTS, 'mean_acc.pickle'), 'wb') as handle:
    pickle.dump(mean_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)



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
        PATH_SAVE=os.path.join(r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion",
        "per_sub.png")
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


def plot_performance_feature_comparison():

    df_sw = pd.read_pickle("df_Sharpwave.p")
    df_fft = pd.read_pickle("df_fft.p")
    df_swfft = pd.read_pickle("df_fft_sw.p")
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

    df_plt = pd.concat(
        [df_comb.query("ch_type == 'all ch combinded'"), df_comb_ind]
    )

    nm_plots.plot_df_subjects(
        df=df_plt,
        y_col="performance_test",
        x_col="ch_type",
        hue="features_used",
        title="best channel performances",
        PATH_SAVE=r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\results_paper_figures_2\LM_comp_per_best_ind.png",
    )


def plt_mean_accuracies_over_time(mean_acc: dict):

    # plot accuracies
    df_acc = pd.DataFrame()

    arr_acc_pls = np.stack([f[0] for f in mean_acc])[:, :10]  # PLS
    arr_acc_unpls = np.stack([f[1] for f in mean_acc])[:, :10]  # UNPLS
    for i in range(arr_acc_pls.shape[0]):
        for t in range(10):
            df_acc = df_acc.append(
                {
                    "sub": i,
                    "Accuracy": arr_acc_pls[i, t],
                    "Stimulus": "Neutral",
                    "Time [ms]": np.round(100 + t * 100),
                },
                ignore_index=True,
            )

            df_acc = df_acc.append(
                {
                    "sub": i,
                    "Accuracy": arr_acc_unpls[i, t],
                    "Stimulus": "pls+unpls",
                    "Time [ms]": np.round(100 + t * 100),
                },
                ignore_index=True,
            )

    nm_plots.plot_df_subjects(
        df=df_acc,
        x_col="Time [ms]",
        y_col="Accuracy",
        title="Averaged Performances Best Channels",
        hue="Stimulus",
        PATH_SAVE=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\results\results_ntr_vs_emotion\acc_comp_sub.png",
    )
 
    sb.pointplot(x="Time [ms]", y="Accuracy", data=df_acc.groupby(["sub", "Stimulus", "Time [ms]"]).median().reset_index(), hue="Stimulus", palette="viridis",
    errwidth=0, capsize=0, markers=[False, False], dodge=True)

    # OPTIONALLY PLOT MEAN LINES
    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(
        np.arange(0, 1, 0.1),
        np.stack([f[2] for f in mean_acc]).mean(axis=0)[:10],
        label="PLS",
    )
    plt.plot(
        np.arange(0, 1, 0.1),
        np.stack([f[3] for f in mean_acc]).mean(axis=0)[:10],
        label="UNPLS",
    )
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Accuracy")
    plt.title("predictions")
    plt.savefig(
        r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\results_paper_figure\temporal_accuracy.png",
    )


def plot_mean_coefficients(df):
    mean_coef_lm = np.abs(
        np.stack(df.query("all_combined == 0")["coef"]).mean(axis=0)[0, :]
    )  # for single class
    feature_names = [
        f[8:] for f in analyzer.decoder.feature_names if f.startswith("Cg25R01")
    ]

    plt.figure(figsize=(15, 10), dpi=300)
    sort_idx = np.argsort(mean_coef_lm)[::-1]
    plt.bar(np.arange(len(feature_names)), mean_coef_lm[sort_idx])
    plt.xticks(
        np.arange(len(feature_names)),
        np.array(feature_names)[sort_idx],
        rotation=90,
    )
    plt.ylabel("Linear Model Mean coefficients")
    plt.title("Mean channel and subject LM coefficients")
    plt.tight_layout()
    plt.savefig(
        r"C:\Users\ICN_admin\Documents\TRD Analysis\30_05\results_paper_figure\LM_coeff_single_ch_mean.png"
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

    #df = pd.read_pickle("df_fft_sw.p")

    df_ind_ch = df.query("ch_type == 'electrode ch'").reset_index()
    df_ind_ch["sub_str"] = df_ind_ch["sub"].str.split("_").apply(lambda x: x[1])
    df_ind_ch["bdi"] = df_ind_ch.sub_str.map(bdi_mapping)
    df_ind_ch["bdi_change"] = df_ind_ch.sub_str.map(bdi_change_mapping)
    idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
    df_mean_sub_ind_ch = df_ind_ch.iloc[idx]

    nm_plots.reg_plot(
        x_col="performance_test", y_col="bdi_change", data=df_mean_sub_ind_ch
    )
