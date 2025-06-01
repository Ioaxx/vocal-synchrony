import pandas as pd
import os
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

def gather_data_for_gender_state(directory, gender, state):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and f"{gender}_{state}" in filename:
            df = pd.read_csv(os.path.join(directory, filename))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def run_anova_per_feature(directory):
    results = {}

    for gender in ['female', 'male']:
        print(f"\nRunning ANOVA for: {gender}")

        normal_df = gather_data_for_gender_state(directory, gender, "normal")
        aroused_df = gather_data_for_gender_state(directory, gender, "aroused")

        common_features = list(set(normal_df.columns) & set(aroused_df.columns))

        gender_results = []

        for feature in common_features:
            # Only run ANOVA on numeric features
            if pd.api.types.is_numeric_dtype(normal_df[feature]):
                # Drop NaNs to be safe
                group1 = normal_df[feature].dropna()
                group2 = aroused_df[feature].dropna()

                if len(group1) > 1 and len(group2) > 1:
                    f_stat, p_val = f_oneway(group1, group2)
                    gender_results.append({
                        'feature': feature,
                        'F_stat': f_stat,
                        'p_value': p_val
                    })

        results[gender] = pd.DataFrame(gender_results).sort_values("p_value")

    return results

def plot_anova_results(df, gender):
    df['significant'] = df['p_value'] < 0.05
    df = df.sort_values(by='F_stat', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='feature', y='F_stat', hue='significant', palette={True: 'tomato', False: 'gray'})

    plt.title(f"ANOVA F-Statistics per Feature ({gender.capitalize()} Aroused vs Normal)")
    plt.ylabel("F-Statistic")
    plt.xlabel("Feature")
    plt.xticks(rotation=45)
    plt.axhline(1, color='black', linestyle='--', linewidth=0.8, label='F=1 reference')
    plt.legend(title="p < 0.05")
    plt.tight_layout()
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{gender}_anova_plot.png"))
    plt.close()

# Usage example:
directory = "./output/processed"  # adjust if needed
anova_results = run_anova_per_feature(directory)

# Plot female results
plot_anova_results(anova_results['female'], 'female')

# Plot male results
plot_anova_results(anova_results['male'], 'male')