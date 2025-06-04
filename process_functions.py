import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from scipy.stats import f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_FOLDER="input/"


couples = [f"couple{i}" for i in range(1, 8)]
genders = ['female', 'male']
states = ['baseline','aroused']

def summarize_voice_features(df):
    feature_groups = {
        'pitch': [col for col in df.columns if 'F0finEnv' in col],
        'loudness': [col for col in df.columns if col.startswith('pcm_loudness_')],
        'jitter': [col for col in df.columns if col.startswith('jitterLocal') or col.startswith('jitterDDP')],
        'shimmer': [col for col in df.columns if col.startswith('shimmerLocal')],
        'speechrate': [col for col in df.columns if 'duration' in col or 'risetime' in col or 'falltime' in col],
        'timbre': [col for col in df.columns if col.startswith('pcm_fftMag_mfcc')],
        'energy': [col for col in df.columns if 'variance' in col],
        'alignment': [col for col in df.columns if any(x in col for x in ['ctime', 'maxPos', 'minPos', 'upleveltime', 'downleveltime'])]
    }
    
    summary = {}
    for group, cols in feature_groups.items():
        valid_cols = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if valid_cols:
            summary[group] = df[valid_cols].mean(axis=1)
        else:
            summary[group] = pd.Series([np.nan]*len(df), index=df.index)
            
    return pd.DataFrame(summary)  # Corrected line


def load_all_data():
    data = {}
    data_mfc = {}
    file={}
    for c in couples:
        data[c] = {}
        data_mfc[c] = {}
        file[c] = {}
        for g in genders:
            data[c][g] = {}
            data_mfc[c][g] = {}
            file[c][g] = {}
            for s in states:
                print(f"Loading {c} {g} {s}")
                data_mfc[c][g][s], file[c][g][s], data[c][g][s] = load_clean(c, g, s)
                file[c][g][s].to_csv(f"./output/raw/{c}_{g}_{s}.csv")
    return data, file, data_mfc


def load_and_save_processed_data():
    for c in couples:
        for g in genders:
            for s in states:
                print(f"Loading and processing: {c} {g} {s}")
                _, _, processed = load_clean(c, g, s)
                
                # Save to file
                out_path = f"./output/processed/{c}_{g}_{s}.csv"
                processed.to_csv(out_path, index=False)


def load_clean(couple,gender,state):
    filename = f"{couple}_{gender}_{state}.csv"
    path = os.path.join(DATA_FOLDER, filename)
    mfcc_vector, relavant_df, summary_df = clean_data(path)
    return mfcc_vector, relavant_df, summary_df


def get_feature_groups(df):
    return {
        'pitch': [col for col in df.columns if col.startswith('F0finEnv_')],
        'loudness': [col for col in df.columns if col.startswith('pcm_loudness_')],
        'jitter': [col for col in df.columns if col.startswith('jitterLocal_') or col.startswith('jitterDDP_')],
        'shimmer': [col for col in df.columns if col.startswith('shimmerLocal_')],
        'speechrate': [col for col in df.columns if 'duration' in col or 'risetime' in col or 'falltime' in col],
        'timbre': [col for col in df.columns if 'mfcc' in col and 'centroid' in col],
        'energy': [col for col in df.columns if 'variance' in col or 'energy' in col],
        'alignment': [col for col in df.columns if any(x in col for x in ['ctime', 'maxPos', 'minPos', 'upleveltime', 'downleveltime'])]
    }


def clean_data(input_file):
    df = pd.read_csv(input_file, sep=';', decimal='.')

    feature_groups = get_feature_groups(df)

    all_relevant_cols = list(set(col for cols in feature_groups.values() for col in cols))
    mfcc_cols = feature_groups['timbre']

    relevant_df = df[all_relevant_cols]
    relevant_df = shuffle(relevant_df).reset_index(drop=True)
    summary_df = summarize_voice_features(df)
    MFCC_vector_df = df[mfcc_cols]

    return MFCC_vector_df, relevant_df, summary_df

def drop_constant_columns(df):
    return df.loc[:, df.apply(lambda col: col.nunique() > 1)]



def mfcc_comparison(input_1, input_2):

    mfcc_cols_1 = [col for col in input_1.columns if 'mfcc' in col]
    mfcc_vector_1 = input_1[mfcc_cols_1].mean().values.reshape(1, -1)


    mfcc_cols_2 = [col for col in input_2.columns if 'mfcc' in col]
    mfcc_vector_2 = input_2[mfcc_cols_2].mean().values.reshape(1, -1)


    similarity = cosine_similarity(mfcc_vector_1, mfcc_vector_2)
    distance = 1 - similarity

    return distance


def t_test(df1, df2):
    results = []
    numeric_cols = df1.select_dtypes(include='number').columns.intersection(df2.select_dtypes(include='number').columns)
    
    for col in numeric_cols:
        t_stat, p_val = ttest_ind(df1[col], df2[col], equal_var=False, nan_policy='omit')
        results.append({'feature': col, 't_stat': t_stat, 'p_value': p_val})
    return pd.DataFrame(results).sort_values('p_value')


def plot_mfcc_distance_changes(results_baseline, results_aroused):
    # Flatten the nested lists ([[value]]) to scalars
    baseline_flat = {k: v[0][0] for k, v in results_baseline.items()}
    aroused_flat = {k: v[0][0] for k, v in results_aroused.items()}

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'couple': list(baseline_flat.keys()) * 2,
        'state': ['baseline'] * len(baseline_flat) + ['aroused'] * len(aroused_flat),
        'mfcc_distance': list(baseline_flat.values()) + list(aroused_flat.values())
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='couple', y='mfcc_distance', hue='state')
    plt.title('MFCC Distance between Male and Female by Couple and State')
    plt.ylabel('MFCC Distance')
    plt.xlabel('Couple')
    plt.legend(title='State')
    plt.tight_layout()
    plt.show()


def restructure_data(raw_data):
    rows = []
    for couple, genders in raw_data.items():
        for gender, states in genders.items():
            for state, df in states.items():
                row = df.iloc[0].to_dict()
                row.update({
                    'couple': couple,
                    'gender': gender,
                    'state': state
                })
                rows.append(row)
    return pd.DataFrame(rows)

def restructure_data_with_markers(raw_data):
    """Convert raw data into DataFrame with vocal markers for statistical testing"""
    rows = []
    for couple, genders in raw_data.items():
        for gender, states in genders.items():
            for state, df in states.items():
                # Compute vocal markers for each row
                for idx, row in df.iterrows():
                    temp_df = pd.DataFrame([row])
                    markers = summarize_voice_features(temp_df).iloc[0].to_dict()
                    
                    rows.append({
                        'couple': couple,
                        'gender': gender,
                        'state': state,
                        **markers
                    })
    return pd.DataFrame(rows)

def run_t_tests(df):
    results = []
    for couple in df['couple'].unique():
        couple_df = df[df['couple'] == couple]
        for attr in ['pitch', 'loudness', 'jitter', 'shimmer', 'speechrate', 'timbre', 'energy', 'alignment']:
            # Male baseline vs Female baseline
            male_baseline = couple_df[(couple_df['gender'] == 'male') & (couple_df['state'] == 'baseline')]
            female_baseline = couple_df[(couple_df['gender'] == 'female') & (couple_df['state'] == 'baseline')]
            if not male_baseline.empty and not female_baseline.empty:
                t_stat, p_val = ttest_ind(male_baseline[attr], female_baseline[attr], equal_var=False)
                results.append({'couple': couple, 'attribute': attr, 'comparison': 'male_baseline_vs_female_baseline', 't_stat': t_stat, 'p_value': p_val})
            # Male aroused vs Female aroused
            male_aroused = couple_df[(couple_df['gender'] == 'male') & (couple_df['state'] == 'aroused')]
            female_aroused = couple_df[(couple_df['gender'] == 'female') & (couple_df['state'] == 'aroused')]
            if not male_aroused.empty and not female_aroused.empty:
                t_stat, p_val = ttest_ind(male_aroused[attr], female_aroused[attr], equal_var=False)
                results.append({'couple': couple, 'attribute': attr, 'comparison': 'male_aroused_vs_female_aroused', 't_stat': t_stat, 'p_value': p_val})
    return pd.DataFrame(results)


def run_anova_tests(df):
    results = []
    attributes = ['pitch', 'loudness', 'jitter', 'shimmer', 'speechrate', 'timbre', 'energy', 'alignment']
    for attr in attributes:
        groups = [df[df['couple'] == c][attr] for c in df['couple'].unique()]
        f_stat, p_val = f_oneway(*groups)
        results.append({'attribute': attr, 'f_stat': f_stat, 'p_value': p_val})
    return pd.DataFrame(results)

def compile_results(t_test_df, anova_df):
    t_test_df['test_type'] = 't_test'
    anova_df['test_type'] = 'anova'
    anova_df['couple'] = None
    anova_df['comparison'] = None
    t_test_df = t_test_df[['couple', 'attribute', 'comparison', 't_stat', 'p_value', 'test_type']]
    anova_df = anova_df[['couple', 'attribute', 'comparison', 'f_stat', 'p_value', 'test_type']]
    combined = pd.concat([t_test_df, anova_df], ignore_index=True)
    return combined

def plot_marker_comparisons(df, marker):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='couple', y=marker, hue='state')
    plt.title(f'Comparison of {marker} by Couple and State')
    plt.ylabel(marker)
    plt.xlabel('Couple')
    plt.legend(title='State')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def create_complete_feature_markers(data):
    """
    Create a DataFrame with all vocal markers from raw data
    """
    marker_rows = []
    required_features = ['pitch', 'loudness', 'jitter', 'shimmer',
                        'speechrate', 'timbre', 'energy', 'alignment']
    
    for couple in data:
        for gender in ['male', 'female']:
            for state in ['baseline', 'aroused']:
                df = data[couple][gender][state]
                
                feature_groups = {
                    'pitch': [col for col in df.columns if 'F0finEnv' in col],
                    'loudness': [col for col in df.columns if col.startswith('pcm_loudness_')],
                    'jitter': [col for col in df.columns if col.startswith('jitterLocal') or col.startswith('jitterDDP')],
                    'shimmer': [col for col in df.columns if col.startswith('shimmerLocal')],
                    'speechrate': [col for col in df.columns if 'duration' in col or 'risetime' in col or 'falltime' in col],
                    'timbre': [col for col in df.columns if col.startswith('pcm_fftMag_mfcc')],
                    'energy': [col for col in df.columns if 'variance' in col],
                    'alignment': [col for col in df.columns if any(x in col for x in ['ctime', 'maxPos', 'minPos', 'upleveltime', 'downleveltime'])]
                }
                
                for idx, row in df.iterrows():
                    row_features = {'couple': couple, 'gender': gender, 'state': state}
                    
                    # Ensure all features get created even if no columns found
                    for feature in required_features:
                        cols = feature_groups.get(feature, [])
                        valid_cols = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                        row_features[feature] = row[valid_cols].mean() if valid_cols else np.nan
                    
                    marker_rows.append(row_features)
    
    return pd.DataFrame(marker_rows)[['couple', 'gender', 'state'] + required_features]

def run_t_tests2(df):
    results = []
    for couple in df['couple'].unique():
        couple_df = df[df['couple'] == couple]
        for state in ['baseline', 'aroused']:
            state_df = couple_df[couple_df['state'] == state]
            male = state_df[state_df['gender'] == 'male']
            female = state_df[state_df['gender'] == 'female']
            
            for attr in ['pitch', 'loudness', 'jitter', 'shimmer', 'speechrate', 'timbre', 'energy', 'alignment']:
                if attr not in df.columns:
                    continue  # Skip missing features
                
                if len(male) > 1 and len(female) > 1:
                    try:
                        t_stat, p_val = ttest_ind(male[attr].dropna(), female[attr].dropna(), equal_var=False)
                        results.append({
                            'couple': couple,
                            'attribute': attr,
                            'comparison': f'male_{state}_vs_female_{state}',
                            't_stat': t_stat,
                            'p_value': p_val,
                            'test_type': 't_test'
                        })
                    except:
                        continue
    return pd.DataFrame(results)

