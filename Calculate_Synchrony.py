import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate
from dtw import dtw
import random

# Parameters - adjust these
data_folder = './data'
couples = ['couple1', 'couple2', 'couple3']  # your couple IDs here
genders = ['male', 'female']
states = ['normal', 'aroused']
feature_name = 'F0finEnv_max'
num_permutations = 1000

def load_feature(filepath, feature=feature_name):
    df = pd.read_csv(filepath, sep=';')
    return df[feature].values

def align_series(x, y):
    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]

def cross_correlation(x, y):
    corr = correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lags = np.arange(-len(x)+1, len(x))
    max_corr = np.max(corr) / (np.std(x)*np.std(y)*len(x))
    max_lag = lags[np.argmax(corr)]
    return max_corr, max_lag

def dtw_distance(x, y):
    dist, _, _, _ = dtw(x, y, dist=lambda a,b: abs(a-b))
    return dist

def permutation_test(x, y, observed_corr, num_perm=num_permutations):
    count = 0
    for _ in range(num_perm):
        y_perm = np.random.permutation(y)
        corr, _ = pearsonr(x, y_perm)
        if abs(corr) >= abs(observed_corr):
            count += 1
    p_val = count / num_perm
    return p_val

results = []

for c in couples:
    for s in states:
        # build file paths
        file_male = os.path.join(data_folder, f"{c}_male_{s}.csv")
        file_female = os.path.join(data_folder, f"{c}_female_{s}.csv")
        
        if not os.path.exists(file_male) or not os.path.exists(file_female):
            print(f"Skipping {c} {s} due to missing files")
            continue
        
        # Load data
        f0_male = load_feature(file_male)
        f0_female = load_feature(file_female)
        
        # Align
        f0_male_al, f0_female_al = align_series(f0_male, f0_female)
        
        # Compute Pearson
        pearson_corr, pearson_p = pearsonr(f0_male_al, f0_female_al)
        
        # Compute cross-correlation max
        max_cross_corr, lag = cross_correlation(f0_male_al, f0_female_al)
        
        # Compute DTW
        dtw_dist = dtw_distance(f0_male_al, f0_female_al)
        
        # Permutation test on Pearson corr
        perm_p = permutation_test(f0_male_al, f0_female_al, pearson_corr)
        
        results.append({
            'couple': c,
            'state': s,
            'pearson_corr': pearson_corr,
            'pearson_pvalue': pearson_p,
            'perm_pvalue': perm_p,
            'max_cross_corr': max_cross_corr,
            'cross_corr_lag': lag,
            'dtw_distance': dtw_dist
        })
        print(f"Processed {c} {s}: Pearson r={pearson_corr:.3f} perm p={perm_p:.3f}")

# Save results CSV
results_df = pd.DataFrame(results)
results_df.to_csv('synchrony_results.csv', index=False)

# Plotting Pearson correlations per state
plt.figure(figsize=(10,5))
for s in states:
    subset = results_df[results_df['state'] == s]
    plt.hist(subset['pearson_corr'], alpha=0.6, bins=10, label=f'Pearson r {s}')
plt.title('Distribution of Pearson Correlations (Male-Female F0)')
plt.xlabel('Pearson Correlation')
plt.ylabel('Count')
plt.legend()
plt.savefig('pearson_correlation_distribution.png')

# Plotting permutation p-values per state
plt.figure(figsize=(10,5))
for s in states:
    subset = results_df[results_df['state'] == s]
    plt.hist(subset['perm_pvalue'], alpha=0.6, bins=10, label=f'Permutation p-value {s}')
plt.title('Permutation Test p-values for Synchrony (Male-Female F0)')
plt.xlabel('Permutation p-value')
plt.ylabel('Count')
plt.legend()
plt.savefig('permutation_pvalue_distribution.png')

print("Done! Results saved to 'synchrony_results.csv' and plots saved as PNG files.")
