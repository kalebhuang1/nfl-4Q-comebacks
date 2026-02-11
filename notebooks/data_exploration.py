import nflreadpy as nfl  # Switched to nflreadpy
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import ssl

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV



plt.style.use('Solarize_Light2')
ssl._create_default_https_context = ssl._create_unverified_context

def load_comeback_data(years=list(range(2015,2025))):
    cache_file = "nfl_pbp_cache.csv"
    
    columns = [
        'game_id', 'season', 'home_team', 'away_team', 'posteam', 'defteam',
        'qtr', 'game_seconds_remaining', 'score_differential',
        'yardline_100', 'down', 'ydstogo',
        'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
        'spread_line', 'epa', 'wind', 'total_line',
        'kick_distance', 'temp', 'goal_to_go', 'result'
    ]

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

    # Enforce the modeling schema for both cached and freshly-downloaded data.
    missing_after_refresh = [c for c in columns if c not in pbp.columns]
    if missing_after_refresh:
        raise ValueError(
            "These required columns are unavailable in the loaded PBP data: "
            + ", ".join(missing_after_refresh)
        )
    pbp = pbp[columns]

    pbp = add_epa(pbp)
    pbp = add_home_team(pbp)
    df = pbp[pbp['qtr'] == 4].copy()

    def determine_win(row):
        if row['posteam'] == row['home_team']:
            return 1 if row['result'] > 0 else 0
        else:
            return 1 if row['result'] < 0 else 0

    df['won_game'] = df.apply(determine_win, axis=1)

    comeback_df = df[(df['qtr'] >= 4) & (df['score_differential'] < 0) & (df['score_differential'] >= -8)].copy()
    comeback_df = comeback_df.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, False])

    # Build drive IDs from possession changes within each game.
    drive_start = comeback_df['posteam'] != comeback_df.groupby('game_id')['posteam'].shift(1)
    comeback_df['drive_id'] = drive_start.groupby(comeback_df['game_id']).cumsum()

    # Keep only each team's final trailing drive in the game.
    last_drive = comeback_df.groupby(['game_id', 'posteam'])['drive_id'].transform('max')
    comeback_df = comeback_df[comeback_df['drive_id'] == last_drive]

    # Keep only the first play of that drive (most time remaining).
    comeback_df = (
        comeback_df.sort_values(['game_id', 'posteam', 'game_seconds_remaining'], ascending=[True, True, False])
        .groupby(['game_id', 'posteam'], as_index=False)
        .head(1)
        .drop(columns=['drive_id'])
    )

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

def add_home_team(df):
    df['is_home'] = (df['posteam'] == df['home_team']).astype(int)
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
    unique_games = df['game_id'].unique()
    train_games, temp_games = train_test_split(unique_games, test_size=0.4, random_state=42)
    val_games, test_games = train_test_split(temp_games, test_size = 0.5, random_state= 42)
    
    train_df = df[df['game_id'].isin(train_games)].copy()
    test_df = df[df['game_id'].isin(test_games)].copy()
    val_df   = df[df["game_id"].isin(val_games)].copy()

    string_columns = train_df.select_dtypes(include=['object']).columns
    train_df = train_df.drop(columns = string_columns)
    test_df = test_df.drop(columns = string_columns)
    val_df = val_df.drop(columns = string_columns)
    X_train = train_df.drop(columns = ['won_game', 'result', 'epa']).fillna(0)
    y_train = train_df['won_game']
    X_test = test_df.drop(columns = ['won_game', 'result', 'epa']).fillna(0)
    y_test = test_df['won_game']
    X_val = val_df.drop(columns = ['won_game', 'result', 'epa']).fillna(0)
    y_val = val_df["won_game"]
    
    return X_train, X_test, y_train, y_test, X_val, y_val

def rf_model(X_train, X_test, y_train, y_test, X_val, y_val):
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
    search = RandomizedSearchCV(
        estimator = rf_base,
        param_distributions= search_space,
        n_iter = 40,
        scoring = 'f1',
        cv = 5,
        random_state=42,
        n_jobs = 1,
        verbose = 1
    )
    search.fit(X_train, y_train)
    print("Best CV F1:", search.best_score_)
    print("Best Params:", search.best_params_)

    best_rf = search.best_estimator_
    rf_model = CalibratedClassifierCV(best_rf, method='sigmoid', cv=5)
    rf_model.fit(X_train, y_train)
    
    val_probs = rf_model.predict_proba(X_val)[:,1]
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

    test_probs = rf_model.predict_proba(X_test)[:,1]
    predictions = (test_probs >= best_t).astype(int)
    acc = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {acc:.2%}') 
    print(classification_report(y_test, predictions))
    return rf_model, best_t

def plot_confusion_matrix(rf_model, X_test, y_test, threshold):
    test_probs = rf_model.predict_proba(X_test)[:,1]
    preds = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Comeback Prediction: True vs Predicted")
    plt.savefig("results/plots/comeback_confusion_matrix.png")

def plot_smart_importance(calibrated_model, X_train):
    importances_list = []

    for clf in calibrated_model.calibrated_classifiers_:

        est = clf.estimator 
        
        if isinstance(est, Pipeline):
            importances_list.append(est.named_steps['clf'].feature_importances_)
        else:
            importances_list.append(est.feature_importances_)
    all_importances = np.array(importances_list)
    importances = np.mean(all_importances, axis=0)
    std = np.std(all_importances, axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns
    plt.figure(figsize=(12, 7))
    plt.title("Universal Feature Importance (Averaged Over Folds)")
    plt.bar(range(X_train.shape[1]), importances[indices], 
            color="skyblue", yerr=std[indices], align="center", capsize=5)
    
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance_universal.png")


df = load_comeback_data()
plot_win_probability_groups(df)

X_train, X_test, y_train, y_test, X_val, y_val = prep_train_data(df)
rf_model, best_t = rf_model(X_train, X_test, y_train, y_test, X_val, y_val)

plot_confusion_matrix(rf_model, X_test, y_test, best_t)
plot_smart_importance(rf_model, X_train)
