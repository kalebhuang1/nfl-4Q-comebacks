import nflreadpy as nfl 
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ssl
import json
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn import set_config
set_config(enable_metadata_routing=True)


plt.style.use('Solarize_Light2')
ssl._create_default_https_context = ssl._create_unverified_context

def load_comeback_data(years=list(range(2015,2025))):
    years_key = "-".join(map(str, sorted(years)))
    cache_file = f"nfl_pbp_cache_{years_key}.csv"
    
    columns = [
        'game_id','qtr', 'home_team', 'away_team', 'posteam', 'defteam','game_seconds_remaining', 'score_differential',
        'yardline_100', 'down', 'ydstogo', 'goal_to_go', 'defteam_timeouts_remaining',
        'posteam_timeouts_remaining', 'epa', 'kick_distance', 'qb_epa', 'posteam_score', 
        'defteam_score', 
         'result', 'season', 'drive', 'drive_play_count'
    ]
#'total_home_comp_air_epa', 'total_away_comp_air_epa',
#'temp', 'wind', 
    def _download_and_cache_full_pbp():
        print(f"Downloading PBP data for years: {years}...")
        full_pbp = nfl.load_pbp(years).to_pandas()
        full_pbp.to_csv(cache_file, index=False)
        print("Data saved to local cache.")
        return full_pbp
    
    if os.path.exists(cache_file):
        print("Loading data from local cache...")
        pbp = pd.read_csv(cache_file, low_memory=False)
        missing_cols = [c for c in columns if c not in pbp.columns]
        if missing_cols:
            print(
                "Cache is missing required columns: "
                + ", ".join(missing_cols)
                + ". Refreshing cache..."
            )
            pbp = _download_and_cache_full_pbp()
    else:
        pbp = _download_and_cache_full_pbp()

    missing_after_refresh = [c for c in columns if c not in pbp.columns]
    if missing_after_refresh:
        raise ValueError(
            "These required columns are unavailable in the loaded PBP data: "
            + ", ".join(missing_after_refresh)
        )
    pbp = pbp[columns]

    pbp = add_epa(pbp)
    pbp = add_home_team(pbp)
    pbp = add_qb_epa(pbp)
    pbp = add_avg_drive_play_count(pbp)

    df = pbp[pbp['qtr'] >= 4].copy()

    def determine_win(row):
        if row['posteam'] == row['home_team']:
            return 1 if row['result'] > 0 else 0
        else:
            return 1 if row['result'] < 0 else 0

    df['won_game'] = df.apply(determine_win, axis=1)

    drive_df = df.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, False]).copy()
    drive_df['is_drive_start'] = (
        drive_df.groupby(['game_id', 'posteam', 'drive']).cumcount() == 0
    )

    comeback_df = drive_df[
        (drive_df['score_differential'] < 0)
        & (drive_df['score_differential'] >= -8)
        & (drive_df['is_drive_start'])
        & (drive_df['down'] == 1)
    ].copy()

    comeback_df = (
        comeback_df.sort_values(['game_id', 'posteam', 'game_seconds_remaining'], ascending=[True, True, True])
        .groupby(['game_id', 'posteam'], as_index=False)
        .tail(1)
        .drop(columns=['is_drive_start'])
    )
    print("after head(1) per team:", comeback_df.shape)
    print(comeback_df.groupby("game_id").size().value_counts().sort_index().head(10))

    return comeback_df

def add_epa(df):
    df = df.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, False], kind='mergesort').copy()

    off_group = df.groupby(['game_id', 'posteam'])['epa']
    off_cumsum = off_group.cumsum()
    off_count_prior = off_group.cumcount()
    df['off_epa_game_prior'] = np.where(
        off_count_prior > 0,
        (off_cumsum - df['epa']) / off_count_prior,
        0.0
    )

    def_group = df.groupby(['game_id', 'defteam'])['epa']
    def_cumsum = def_group.cumsum()
    def_count_prior = def_group.cumcount()
    df['def_epa_game_prior'] = np.where(
        def_count_prior > 0,
        (def_cumsum - df['epa']) / def_count_prior,
        0.0
    )
    return df
def add_qb_epa(df):
    df = df.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, False], kind='mergesort').copy()

    off_group = df.groupby(['game_id', 'posteam'])['qb_epa']
    off_cumsum = off_group.cumsum()
    off_count_prior = off_group.cumcount()
    df['qb_epa_prior'] = np.where(
        off_count_prior > 0,
        (off_cumsum - df['qb_epa']) / off_count_prior,
        0.0
    )

    return df

def add_home_team(df):
    df['is_home'] = (df['posteam'] == df['home_team']).astype(int)
    return df

def add_avg_drive_play_count(df):
    drive_df = (
        df.sort_values(['game_id', 'posteam', 'drive', 'game_seconds_remaining'], ascending=[True, True, True, False])
        .drop_duplicates(subset=['game_id', 'posteam', 'drive'])[['game_id', 'posteam', 'drive', 'drive_play_count']]
    )
    drive_df = drive_df.sort_values(['game_id', 'posteam', 'drive']).copy()

    grp = drive_df.groupby(['game_id', 'posteam'])['drive_play_count']
    cumsum = grp.cumsum()
    count_prior = grp.cumcount()

    drive_df['off_avg_drive_play_count_prior'] = np.where(count_prior > 0,(cumsum - drive_df['drive_play_count']) / count_prior, 0.0)
    df = df.merge(
    drive_df[['game_id', 'posteam', 'drive', 'off_avg_drive_play_count_prior']],
    on=['game_id', 'posteam', 'drive'],
    how='left'
)

    df['off_avg_drive_play_count_prior'] = df['off_avg_drive_play_count_prior'].fillna(0.0)
    return df


def plot_win_probability_groups(df):
    win_rate_groups = df.groupby('score_differential')['won_game'].mean().reset_index().rename(columns = {'won_game': 'win_percent'})
    x = win_rate_groups['score_differential']
    y = win_rate_groups['win_percent']
    plt.figure()
    plt.xlabel('Score Differential')
    plt.ylabel('Win Rate')
    plt.title("4th Quarter Comeback Rate by Score", fontsize=15)
    plt.scatter(x, y)
    
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/01_win_rate_by_points_down.png")

def prep_train_data(df):
    train_df = df[df['season'] <= 2021].copy()
    val_df   = df[(df["season"] >= 2022) & (df["season"] <= 2023)].copy()
    test_df  = df[df["season"] == 2024].copy()

    train_groups = train_df['game_id'].copy()

    string_columns = train_df.select_dtypes(include=['object']).columns
    train_df = train_df.drop(columns=string_columns)
    test_df  = test_df.drop(columns=string_columns)
    val_df   = val_df.drop(columns=string_columns)

    X_train = train_df.drop(columns=['won_game', 'result', 'epa', 'qb_epa','season', 'drive', 'drive_play_count', 'qtr', 'goal_to_go']).fillna(0)
    y_train = train_df['won_game']
    X_test  = test_df.drop(columns=['won_game', 'result', 'epa', 'qb_epa','season', 'drive', 'drive_play_count','qtr', 'goal_to_go']).fillna(0)
    y_test  = test_df['won_game']
    X_val   = val_df.drop(columns=['won_game', 'result', 'epa', 'qb_epa','season', 'drive', 'drive_play_count','qtr', 'goal_to_go']).fillna(0)
    y_val   = val_df['won_game']

    return X_train, X_test, y_train, y_test, X_val, y_val, train_groups


def train_rf_model(X_train, X_test, y_train, y_test, X_val, y_val, train_groups):
    rf_base = RandomForestClassifier(
        random_state=42
    )
    search_space = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 4, 6, 8, 10, 14],
        "min_samples_split": [2,5,10,20],
        "min_samples_leaf": [1,2,4,8],
        "max_features": ["sqrt", "log2", 0.5, 0.7],
        "class_weight": [None, "balanced", "balanced_subsample"]
    }
    gkf = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        estimator = rf_base,
        param_distributions= search_space,
        n_iter = 40,
        scoring = 'f1',
        cv = gkf,
        random_state=42,
        n_jobs = 1,
        verbose = 1
    )
    search.fit(X_train, y_train, groups=train_groups)
    print("Best CV F1:", search.best_score_)
    print("Best Params:", search.best_params_)

    best_rf = search.best_estimator_
    calibrated_rf = CalibratedClassifierCV(best_rf, method='sigmoid', cv=gkf)
    calibrated_rf.fit(X_train, y_train, groups=train_groups)
    
    val_probs = calibrated_rf.predict_proba(X_val)[:,1]
    thresholds = np.arange(0.05, 0.98, 0.02)
    best_t = 0.2
    best_r = -1
    best_p = 0
    best_f1 = 0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        r = recall_score(y_val, preds)
        p = precision_score(y_val, preds, zero_division = 0)
        f = f1_score(y_val, preds)
        if f>best_f1:
            best_r = r
            best_p = p
            best_t = t
            best_f1 = f
    print("Best Threshold: " + str(best_t))
    print("Best Recall: " + str(best_r))
    print("Best Precision: " + str(best_p))
    print("Best F1-Score: " + str(best_f1))

    test_probs = calibrated_rf.predict_proba(X_test)[:,1]
    predictions = (test_probs >= best_t).astype(int)
    acc = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {acc:.2%}') 
    print(classification_report(y_test, predictions))
    return calibrated_rf, best_t

def train_lightgbm_model(X_train, X_test, y_train, y_test, X_val, y_val, train_groups):

    lightgbm_base = LGBMClassifier(objective = 'binary', random_state=42, verbose = -1)
    rs_params={
        "n_estimators": [80, 120, 180, 240],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "num_leaves": [7, 15, 31],
        "max_depth": [3, 4, 5, 6],
        "min_child_samples": [30, 50, 80, 120],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.5, 1.0, 2.0, 5.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0]


    }
    gkf = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        estimator = lightgbm_base,
        param_distributions= rs_params,
        n_iter = 40,
        scoring = 'f1',
        cv = gkf,
        random_state=42,
        n_jobs = 1,
        verbose = 1
    )
    search.fit(X_train, y_train, groups = train_groups)
    print("Best CV F1:", search.best_score_)
    print("Best Params:", search.best_params_)
    best_lgbm = search.best_estimator_

    
    calibrated_rf = CalibratedClassifierCV(best_lgbm, method='sigmoid', cv=gkf)
    calibrated_rf.fit(X_train, y_train, groups = train_groups)

    val_probs = calibrated_rf.predict_proba(X_val)[:,1]
    thresholds = np.arange(0.05, 0.98, 0.01)
    fp_penalties = np.arange(0.45, 1.0, 0.05)
    precision_floors = np.arange(0.3, 0.6, 0.02)
    
    best = {
    "threshold": 0.5,
    "precision_floor": None,
    "fp_penalty": None,
    "f1": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "fp_rate": 0.0,
    }

    for pf in precision_floors:
        for pen in fp_penalties:
            for t in thresholds:
                preds = (val_probs >= t).astype(int)

                p = precision_score(y_val, preds, zero_division=0)
                r = recall_score(y_val, preds)
                f = f1_score(y_val, preds)

                tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
                fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                if p < pf:
                    continue
                if fp_rate > 0.5:
                    continue
    
                if f > best["f1"]:
                    best.update({
                    "threshold": t,
                    "precision_floor": pf,
                    "fp_penalty": pen,
                    "f1": f,
                    "precision": p,
                    "recall": r,
                    "fp_rate": fp_rate,
                    })

    if best["precision_floor"] is None:
        best_f1 = -1
        best_t = 0.5
        for t in thresholds:
            preds = (val_probs >= t).astype(int)
            f = f1_score(y_val, preds)
            if f > best_f1:
                best_f1 = f
                best_t = t
        best["threshold"] = best_t
        best["f1"] = best_f1

    print("Best settings:", best)
    best_t = best["threshold"]


    test_probs = calibrated_rf.predict_proba(X_test)[:,1]
    predictions = (test_probs >= best_t).astype(int)
    acc = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {acc:.2%}') 
    print(classification_report(y_test, predictions))
    return calibrated_rf, best_t




def plot_confusion_matrix(calibrated_rf, X_test, y_test, threshold):
    test_probs = calibrated_rf.predict_proba(X_test)[:,1]
    preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Comeback Prediction: True vs Predicted")
    plt.savefig("results/plots/comeback_confusion_matrix.png")

def plot_smart_importance(calibrated_model, X_train):
    importances_list = []

    if hasattr(calibrated_model, 'calibrated_classifiers_'):
        estimators = [clf.estimator for clf in calibrated_model.calibrated_classifiers_]
    else:
        estimators = [calibrated_model]

    for est in estimators:
        if hasattr(est, 'feature_importances_'):
            importances_list.append(est.feature_importances_)

    if not importances_list:
        raise ValueError("No feature_importances_ found on the fitted estimator(s).")

    all_importances = np.array(importances_list)
    importances = np.mean(all_importances, axis=0)
    std = np.std(all_importances, axis=0)
    feature_names = X_train.columns
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature importance length ({len(importances)}) does not match number of features ({len(feature_names)})."
        )
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 7))
    plt.title("Universal Feature Importance (Averaged Over Folds)")
    plt.bar(range(len(feature_names)), importances[indices], 
            color="skyblue", yerr=std[indices], align="center", capsize=5)
    
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance_universal.png")

def save_baseline_model(model, threshold, feature_names, model_name="lgbm_calibrated_baseline", reports=None):
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/{model_name}_{timestamp}.joblib"
    meta_path = f"models/{model_name}_{timestamp}.json"

    joblib.dump(model, model_path)
    metadata = {
        "saved_at": timestamp,
        "model_name": model_name,
        "threshold": float(threshold),
        "feature_names": list(feature_names)
    }
    if reports is not None:
        metadata["classification_reports"] = reports
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")

def analyze_errors(model, X, y, threshold, split_name="test"):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    eval_df = X.copy()
    eval_df["y_true"] = y.to_numpy()
    eval_df["y_pred"] = preds
    eval_df["proba"] = probs

    eval_df["error_type"] = "correct"
    eval_df.loc[(eval_df["y_true"] == 1) & (eval_df["y_pred"] == 0), "error_type"] = "FN"
    eval_df.loc[(eval_df["y_true"] == 0) & (eval_df["y_pred"] == 1), "error_type"] = "FP"

    print(f"\n{split_name.upper()} error counts")
    print(eval_df["error_type"].value_counts())

    if "game_seconds_remaining" in eval_df.columns:
        eval_df["time_bin"] = pd.cut(
            eval_df["game_seconds_remaining"],
            bins=[0, 120, 300, 600, 900, 3600],
            labels=["0-2m", "2-5m", "5-10m", "10-15m", "15m+"],
            include_lowest=True
        )
        print(f"\n{split_name.upper()} error share by time_bin")
        print(pd.crosstab(eval_df["time_bin"], eval_df["error_type"], normalize="index"))

    if "score_differential" in eval_df.columns:
        eval_df["score_bin"] = pd.cut(
            eval_df["score_differential"],
            bins=[-9, -6, -4, -2, 0],
            labels=["-8:-7", "-6:-5", "-4:-3", "-2:-1"],
            include_lowest=True
        )
        print(f"\n{split_name.upper()} error share by score_bin")
        print(pd.crosstab(eval_df["score_bin"], eval_df["error_type"], normalize="index"))

    if "down" in eval_df.columns:
        print(f"\n{split_name.upper()} error share by down")
        print(pd.crosstab(eval_df["down"], eval_df["error_type"], normalize="index"))

    return eval_df


df = load_comeback_data()
plot_win_probability_groups(df)

X_train, X_test, y_train, y_test, X_val, y_val, train_groups = prep_train_data(df)
calibrated_lgbm, best_t = train_lightgbm_model(X_train, X_test, y_train, y_test, X_val, y_val, train_groups)

plot_confusion_matrix(calibrated_lgbm, X_test, y_test, best_t)
plot_smart_importance(calibrated_lgbm, X_train)

train_probs = calibrated_lgbm.predict_proba(X_train)[:, 1]
train_preds = (train_probs >= best_t).astype(int)

print(classification_report(y_train, train_preds))

val_probs = calibrated_lgbm.predict_proba(X_val)[:,1]
val_preds = (val_probs>=best_t).astype(int)

print(classification_report(y_val, val_preds))
test_probs = calibrated_lgbm.predict_proba(X_test)[:,1]
test_preds = (test_probs>=best_t).astype(int)

reports = {
    "train": classification_report(y_train, train_preds, output_dict=True),
    "val": classification_report(y_val, val_preds, output_dict=True),
    "test": classification_report(y_test, test_preds, output_dict=True)
}
save = False
if save == True:
    save_baseline_model(calibrated_lgbm, best_t, X_train.columns, reports=reports)

error_df = analyze_errors(calibrated_lgbm, X_test, y_test, best_t, split_name="test")
