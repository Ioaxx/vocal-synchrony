import pandas as pd
import os
from scipy.stats import ttest_ind

def gather_data_for_gender_state(directory, gender, state):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and f"{gender}_{state}" in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def run_ttests_per_feature(directory):
    results = {}

    for gender in ['female', 'male']:
        print(f"\nRunning t-tests for: {gender}")

        # Load all files across couples
        normal_df = gather_data_for_gender_state(directory, gender, "normal")
        aroused_df = gather_data_for_gender_state(directory, gender, "aroused")

        # Ensure both datasets have the same columns
        common_features = list(set(normal_df.columns) & set(aroused_df.columns))

        gender_results = []

        for feature in common_features:
            if pd.api.types.is_numeric_dtype(normal_df[feature]):
                stat, p = ttest_ind(aroused_df[feature], normal_df[feature], equal_var=False, nan_policy='omit')
                gender_results.append({
                    'feature': feature,
                    't_stat': stat,
                    'p_value': p
                })

        # Sort by p-value
        results[gender] = pd.DataFrame(gender_results).sort_values("p_value")

    return results