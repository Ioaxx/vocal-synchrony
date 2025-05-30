# Here's the optimized version - replaces the problematic parts
from process_functions import *
import pandas as pd
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load data with progress tracking
    print("Loading data...")
    data_summarize, data, data_mfc = load_all_data()
    
    # MFCC Analysis - optimized
    print("Computing MFCC distances...")
    results_normal = {}
    results_aroused = {}
    
    for i in range(1, 8):
        key = f"couple{i}"
        # Process male vs female comparisons
        results_normal[key] = mfcc_comparison(
            data_mfc[key]["male"]["normal"], 
            data_mfc[key]["female"]["normal"]
        )[0][0]  # Directly extract scalar value
        print(f"Couple {key} normal: {results_normal[key]:.2e}")

        results_aroused[key] = mfcc_comparison(
            data_mfc[key]["male"]["aroused"],
            data_mfc[key]["female"]["aroused"] 
        )[0][0]
        print(f"Couple {key} aroused: {results_aroused[key]:.2e}")

    # Create DataFrames for statistical testing
    df_normal = pd.DataFrame({'mfcc_distance': results_normal})
    df_aroused = pd.DataFrame({'mfcc_distance': results_aroused})
    
    # Perform t-test
    results = t_test(df_normal, df_aroused)
    print(f"MFCC t-test results:\n{results}")

    # Create markers dataset using vectorized operations
    print("Creating feature markers...")
    marker_dfs = []
    
    for couple in data:
        for gender in ['male', 'female']:
            for state in ['normal', 'aroused']:
                df = data[couple][gender][state]
                # Vectorized feature calculation
                features = pd.DataFrame({
                    'pitch': df[[col for col in df.columns if 'F0finEnv' in col]].mean(axis=1),
                    'loudness': df[[col for col in df.columns if 'pcm_loudness' in col]].mean(axis=1),
                    # Add other features similarly...
                })
                features['couple'] = couple
                features['gender'] = gender
                features['state'] = state
                marker_dfs.append(features)
    
    df_with_markers = pd.concat(marker_dfs, ignore_index=True)
    df_with_markers = create_complete_feature_markers(data)
    # Statistical analysis
    print("Running statistical tests...")
    t_test_df = run_t_tests2(df_with_markers)
    anova_df = run_anova_tests(df_with_markers)
    
    # Save results
    results_df = compile_results(t_test_df, anova_df)
    os.makedirs("./output/results text", exist_ok=True)
    results_df.to_csv("./output/results text/results.csv")
    print("Results saved to CSV")

    # Visualization
    print("Creating visualizations...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, feature in enumerate(['pitch', 'loudness', 'jitter', 'shimmer', 
                                  'speechrate', 'timbre', 'energy', 'alignment']):
        ax = axes[idx//4, idx%4]
        sns.boxplot(data=df_with_markers, x='couple', y=feature, hue='state', ax=ax)
        ax.set_title(feature.capitalize())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("./output/vocal_features.png", dpi=300)
    print("Visualization saved to output/vocal_features.png")

if __name__ == "__main__":
    main()
