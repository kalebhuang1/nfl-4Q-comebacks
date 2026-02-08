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

plt.style.use('Solarize_Light2')
ssl._create_default_https_context = ssl._create_unverified_context

def load_comeback_data(years=list(range(2015,2025))):
    cache_file = "nfl_pbp_cache.csv"
    
    columns = [
        'game_id', 'home_team', 'away_team', 'posteam', 'defteam', 'qtr', 
        'score_differential', 'game_seconds_remaining', 'yardline_100', 
        'down', 'ydstogo', 'play_type', 'result', 'total_home_score', 'total_away_score', 'epa', 'season',
        'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'spread_line'
    ]
    
    if os.path.exists(cache_file):
        print("Loading data from local cache...")
        pbp = pd.read_csv(cache_file, low_memory=False)
    else:
        print(f"Downloading PBP data for years: {years}...")
        pbp = nfl.load_pbp(years).to_pandas()
        pbp = pbp[columns] 
        pbp.to_csv(cache_file, index=False)
        print("Data saved to local cache.")

    pbp = add_epa(pbp)
    pbp = add_home_team(pbp)
    df = pbp[pbp['qtr'] == 4].copy()

    def determine_win(row):
        if row['posteam'] == row['home_team']:
            return 1 if row['result'] > 0 else 0
        else:
            return 1 if row['result'] < 0 else 0

    df['won_game'] = df.apply(determine_win, axis=1)

    comeback_df = df[(df['qtr'] >= 4) & (df['score_differential'] < 0) & (df['score_differential']>=-8)].copy()
    comeback_df = comeback_df.sort_values(['game_id', 'game_seconds_remaining'], ascending=[True, False])
    comeback_df = comeback_df[
        (comeback_df['posteam'] != comeback_df['posteam'].shift(1)) | 
        (comeback_df['down'] == 3) |(comeback_df['down'] == 4)
    ]

    return comeback_df

def add_epa(df):
    off_epa = df.groupby(['season', 'posteam'])['epa'].mean().reset_index()
    off_epa.columns = ['season', 'posteam', 'off_epa_season']

    def_epa = df.groupby(['season', 'defteam'])['epa'].mean().reset_index()
    def_epa.columns = ['season', 'defteam', 'def_epa_season']

    df = df.merge(off_epa, on=['season', 'posteam'], how='left')
    df = df.merge(def_epa, on=['season', 'defteam'], how='left')

    df['off_epa_season'] = df['off_epa_season'].fillna(0)
    df['def_epa_season'] = df['def_epa_season'].fillna(0)
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
    train_games, test_games = train_test_split(unique_games, test_size=0.25, random_state=42)
    
    train_df = df[df['game_id'].isin(train_games)]
    test_df = df[df['game_id'].isin(test_games)]
    
    features = ['score_differential', 'game_seconds_remaining', 'yardline_100', 'off_epa_season', 
                'def_epa_season', 'is_home','posteam_timeouts_remaining', 'defteam_timeouts_remaining'
                ,'spread_line','down', 'ydstogo']
    
    X_train = train_df[features].fillna(0)
    y_train = train_df['won_game']
    X_test = test_df[features].fillna(0)
    y_test = test_df['won_game']
    
    return X_train, X_test, y_train, y_test

def test_model(X_train, X_test, y_train, y_test):
    rf_base = RandomForestClassifier(
        n_estimators=100, 
        max_depth=6, 
        class_weight='balanced', 
        min_samples_split=5, 
        random_state=42
    )
    rf_model = CalibratedClassifierCV(rf_base, method='sigmoid', cv=5)
    rf_model.fit(X_train, y_train)
    win_probs = rf_model.predict_proba(X_test)[:,1]
    custom_threshold = 0.4
    predictions = (win_probs >= custom_threshold).astype(int)
    acc = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {acc:.2%}') 
    print(classification_report(y_test, predictions))
    return rf_model

def plot_confusion_matrix(rf_model, X_test, y_test):
    cm = confusion_matrix(y_test, rf_model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Comeback Prediction: True vs Predicted")
    plt.savefig("results/plots/comeback_confusion_matrix.png")

def plot_smart_importance(rf_model, X_train):
    all_importances = np.array([
        clf.estimator.feature_importances_ 
        for clf in rf_model.calibrated_classifiers_
    ])

    importances = np.mean(all_importances, axis=0)
    std = np.std(all_importances, axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns
    plt.figure(figsize=(12, 7))
    plt.title("Feature Importances: Averaged Across Calibrated Folds")
    plt.bar(range(X_train.shape[1]), importances[indices], 
            color="skyblue", yerr=std[indices], align="center", capsize=5)
    
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance_variance.png")


df = load_comeback_data()
plot_win_probability_groups(df)

X_train, X_test, y_train, y_test = prep_train_data(df)
rf_model = test_model(X_train, X_test, y_train, y_test)

plot_confusion_matrix(rf_model, X_test, y_test)
plot_smart_importance(rf_model, X_train)