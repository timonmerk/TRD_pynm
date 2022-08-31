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

PATH_FEATURES = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\features_computed"

PLS = -2
UNPLS = -1
NTR = -3
ALL = -4

LABEL = ALL


def get_idx_select(
    analyzer,
    NTR_VS_EMOTION,
    PLS_VS_UNPLS,
    PLS_VS_UNPLS_VS_NTR,
    PLS_VS_UNPLS_VS_NTR_VS_REST,
    NTR_VS_PLS,
    NTR_VS_UNPLS,
):
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

    elif NTR_VS_PLS is True:
        # no unpls
        # 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
        idx_select = (analyzer.feature_arr["ALL"] != 0) & (
            analyzer.feature_arr["ALL"] != 3
        )
        analyzer.label = np.array(
            analyzer.feature_arr.loc[
                idx_select,
                :,
            ].iloc[:, LABEL]
        )  # keep only PLS and UNPLS labels
        analyzer.label[np.where(analyzer.label == 2)] = 0  # PLS
        analyzer.label[np.where(analyzer.label != 0)] = 1  # NTR

    elif NTR_VS_UNPLS is True:
        # no unpls
        # 0 - REST, 1 - NTR, 2 - PLS, 3 - UNPLS
        idx_select = (analyzer.feature_arr["ALL"] != 0) & (
            analyzer.feature_arr["ALL"] != 2
        )
        analyzer.label = np.array(
            analyzer.feature_arr.loc[
                idx_select,
                :,
            ].iloc[:, LABEL]
        )  # keep only PLS and UNPLS labels
        analyzer.label[np.where(analyzer.label == 3)] = 0  # PLS
        analyzer.label[np.where(analyzer.label != 0)] = 1  # NTR
    return idx_select, analyzer


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


FFT_ONLY = False
SW_ONLY = False
FFT_SW_ONLY = True

SW_Features_Select = [
    "Sharpwave_Max_prominence_range_5_80",
    "Sharpwave_Max_prominence_range_5_30",
    "Sharpwave_Mean_interval_range_5_30",
    "Sharpwave_Mean_interval_range_5_80",
    "Sharpwave_Max_sharpness_range_5_30",
    "Mean_decay_time_range_5_30",
    "Mean_rise_time_range_5_30",
]


def get_analyzer_3500(
    sub,
    NTR_VS_EMOTION,
    PLS_VS_UNPLS,
    PLS_VS_UNPLS_VS_NTR,
    PLS_VS_UNPLS_VS_NTR_VS_REST,
    NTR_VS_PLS,
    NTR_VS_UNPLS,
):
    analyzer = nm_analysis.Feature_Reader(
        feature_dir=PATH_FEATURES,
        feature_file=sub,
        binarize_label=False,
    )

    label_new = np.copy(analyzer.label)
    jumps = np.diff(analyzer.label)
    idx_fix = np.where(jumps > 0)[0]
    for idx in idx_fix:
        for i in range(25):
            label_new[idx + 10 + i] = jumps[idx]
    analyzer.feature_arr["ALL"] = label_new

    idx_select, analyzer = get_idx_select(
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
        map(fix_name_columns, analyzer.feature_arr.columns)
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
        use_nested_cv=True,
        VERBOSE=False,
        undersampling=False,
        oversampling=True,
        mrmr_select=False,
        cca=False,
        pca=False,
        model_save=True,
    )

    return decoder
