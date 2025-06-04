from functions_process_data import *  # Import custom data processing functions
from functions_graphs import *        # Import custom graphing/plotting functions
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, f_oneway, ttest_1samp  # Common statistical tests
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols  # For ANOVA using linear models

# Perform two-way ANOVA on a set of features
def anova_summary(df, features, formula_base="C(state) * C(sex)", alpha=0.05):
    results = []

    for feature in features:
        try:
            # Define the model formula (e.g., "pitch__F0finEnv_sma ~ C(state) * C(sex)")
            formula = f"{feature} ~ {formula_base}"
            model = ols(formula, data=df).fit()  # Fit ordinary least squares model
            anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA

            # Iterate over effects (main and interaction)
            for effect, row in anova_table.iterrows():
                pval = row["PR(>F)"]
                results.append({
                    "feature": feature,
                    "effect": effect,
                    "p_value": pval,
                    "significant": pval < alpha
                })
        except Exception as e:
            print(f"❌ Failed on feature: {feature} — {e}")
    
    return pd.DataFrame(results)  # Return results as a DataFrame


# Perform paired t-tests comparing aroused vs. baseline state for pitch features
def ttest_arousal_vs_baseline_for_pitch(df, pitch_cols):
    results = []
    # Group by couple and sex to perform paired comparisons correctly
    grouped = df.groupby(['couple_id', 'sex'])
    
    for col in pitch_cols:
        baseline_means = []
        aroused_means = []

        for (cid, sex), group in grouped:
            # Extract pitch values for each emotional state
            baseline_vals = group[group['state'] == 'baseline'][col].dropna()
            aroused_vals = group[group['state'] == 'aroused'][col].dropna()
            
            # Skip if either state is missing data
            if len(baseline_vals) == 0 or len(aroused_vals) == 0:
                continue
            
            # Compute mean per state per couple+sex group
            baseline_mean = baseline_vals.mean()
            aroused_mean = aroused_vals.mean()
            
            baseline_means.append(baseline_mean)
            aroused_means.append(aroused_mean)
        
        # Perform paired t-test if enough pairs exist
        if len(baseline_means) >= 2 and len(aroused_means) >= 2:
            t_stat, p_val = ttest_rel(aroused_means, baseline_means)
            results.append({
                'feature': col,
                't_stat': t_stat,
                'p_value': p_val,
                'n_pairs': len(baseline_means)
            })
    
    return pd.DataFrame(results)


# Flatten cell contents if they contain lists or tuples; keep first element
def simplify_cells(df):
    def unpack_if_iterable(x):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return x[0]
        return x
    return df.apply(lambda col: col.map(unpack_if_iterable))
