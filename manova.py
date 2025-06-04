import pandas as pd
import os
from statsmodels.multivariate.manova import MANOVA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def gather_data_for_gender(directory, gender):
    dfs = []
    for state in ['baseline', 'aroused']:
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and f"{gender}_{state}" in filename:
                df = pd.read_csv(os.path.join(directory, filename))
                df['state'] = state
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def run_manova(directory, gender):
    df = gather_data_for_gender(directory, gender)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    formula = ' + '.join(numeric_cols) + ' ~ state'

    maov = MANOVA.from_formula(formula, data=df)
    print(f"MANOVA results for {gender}:\n")
    print(maov.mv_test())
    return df, numeric_cols

def plot_pca(df, numeric_cols, gender):
    pca = PCA(n_components=2)
    X = df[numeric_cols].fillna(0)  # Fill NaNs just in case

    pcs = pca.fit_transform(X)
    df['PC1'] = pcs[:,0]
    df['PC2'] = pcs[:,1]

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='state', style='state', s=60, palette='Set1')
    plt.title(f'PCA plot of {gender.capitalize()} features by state')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.legend(title='State')
    plt.tight_layout()
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{gender}_manova_plot.png"))
    plt.close()

# Directory where CSVs are stored
directory = "./output/processed"

# Female data and plots
df_female, features_female = run_manova(directory, 'female')
plot_pca(df_female, features_female, 'female')

# Male data and plots
df_male, features_male = run_manova(directory, 'male')
plot_pca(df_male, features_male, 'male')
