def anova_vs_control(control_df, *other_dfs, alpha=0.05):

    shared_cols = set(control_df.columns)
    for df in other_dfs:
        shared_cols &= set(df.columns)
    shared_cols = sorted(shared_cols)

    results = []

    for col in shared_cols:
        try:

            groups = [control_df[col].dropna()] + [df[col].dropna() for df in other_dfs]

            f_stat, p_val = f_oneway(*groups)
            results.append({
                'feature': col,
                'F_stat': f_stat,
                'p_value': p_val,
                'significant_vs_control': p_val < alpha
            })
        except Exception as e:
            results.append({
                'feature': col,
                'F_stat': None,
                'p_value': None,
                'significant_vs_control': False,
                'error': str(e)
            })

    return pd.DataFrame(results).sort_values('p_value')



def ttest_vs_control(control_df, *other_dfs, alpha=0.05):
    shared_cols = set(control_df.columns)
    for df in other_dfs:
        shared_cols &= set(df.columns)
    shared_cols = sorted(shared_cols)

    results = []

    for col in shared_cols:
        try:
            control = control_df[col].dropna()
            for i, df in enumerate(other_dfs, start=1):
                test_group = df[col].dropna()
                t_stat, p_val = ttest_ind(control, test_group, equal_var=False)
                results.append({
                    'feature': col,
                    'group': f'group_{i}',
                    't_stat': t_stat,
                    'TTEST_p_value': p_val,
                    'TTEST_significant': p_val < alpha
                })
        except Exception as e:
            results.append({
                'feature': col,
                'group': f'error',
                't_stat': None,
                'TTEST_p_value': None,
                'TTEST_significant': False,
                'error': str(e)
            })

    return pd.DataFrame(results)


def compare_with_control(control_df, *other_dfs, alpha=0.05):
    anova_df = anova_vs_control(control_df, *other_dfs, alpha=alpha)
    ttest_df = ttest_vs_control(control_df, *other_dfs, alpha=alpha)

    # Aggregate t-test results by taking the worst-case (min p-value across test groups)
    ttest_agg = ttest_df.groupby('feature').agg({
        't_stat': 'mean',
        'TTEST_p_value': 'min',
        'TTEST_significant': 'max'  # if any test group was significant
    }).reset_index()

    # Merge both results
    final_df = pd.merge(anova_df, ttest_agg, on='feature', how='outer')

    # Sort by significance (prioritize features with low p-values)
    return final_df.sort_values(by=['ANOVA_p_value', 'TTEST_p_value'], na_position='last')


def compare_feature_changes(df_base1, df_aroused1, df_base2, df_aroused2):
    delta_1 = (df_aroused1.mean() - df_base1.mean()).abs()
    delta_2 = (df_aroused2.mean() - df_base2.mean()).abs()
    diff_between_couples = (delta_1 - delta_2).abs()
    
    result = pd.DataFrame({
        'feature': delta_1.index,
        'delta_couple1': delta_1.values,
        'delta_couple2': delta_2.values,
        'difference': diff_between_couples.values
    }).sort_values('difference', ascending=False)
    
    return result



def run_grouped_anova(df1, df2, group_dict):
    results = []

    for group_name, columns in group_dict.items():
        shared_cols = [col for col in columns if col in df1.columns and col in df2.columns]
        if not shared_cols:
            continue

        # Combine group values into single list per dataframe
        data1 = df1[shared_cols].values.flatten()
        data2 = df2[shared_cols].values.flatten()

        # Remove NaNs
        data1 = data1[~pd.isnull(data1)]
        data2 = data2[~pd.isnull(data2)]

        if len(data1) > 1 and len(data2) > 1:
            f_stat, p_val = f_oneway(data1, data2)
            results.append({
                'group': group_name,
                'f_stat': f_stat,
                'p_value': p_val
            })

    return pd.DataFrame(results).sort_values('p_value')


def analyze_normal_vs_aroused(data):
    print("\n--- normal vs Aroused per gender and couple ---")
    ttest_results = {}
    anova_results = {}
    mfcc_results = {}

    for c in couples:
        ttest_results[c] = {}
        anova_results[c] = {}
        mfcc_results[c] = {}
        for g in genders:
            base_df = data[c][g]['normal']
            aroused_df = data[c][g]['aroused']

            ttest_results[c][g] = t_test(base_df, aroused_df)
            anova_results[c][g] = anova(base_df, aroused_df)
            mfcc_results[c][g] = mfcc_comparison(base_df, aroused_df)

            print(f"{c} {g} normal vs aroused MFCC cosine distance: {mfcc_results[c][g][0][0]:.4f}" if hasattr(mfcc_results[c][g], '__getitem__') else f"{c} {g} normal vs aroused MFCC cosine distance: {mfcc_results[c][g]:.4f}")

    return ttest_results, anova_results, mfcc_results

def analyze_male_vs_female(data):
    print("\n--- Male vs Female per couple and state ---")
    ttest_results = {}
    anova_results = {}
    mfcc_results = {}

    for c in couples:
        ttest_results[c] = {}
        anova_results[c] = {}
        mfcc_results[c] = {}
        for s in states:
            male_df = data[c]['male'][s]
            female_df = data[c]['female'][s]

            ttest_results[c][s] = t_test(male_df, female_df)
            anova_results[c][s] = anova(male_df, female_df)
            mfcc_results[c][s] = mfcc_comparison(male_df, female_df)

            print(f"{c} male vs female in {s} MFCC cosine distance: {mfcc_results[c][s][0][0]:.4f}" if hasattr(mfcc_results[c][s], '__getitem__') else f"{c} male vs female in {s} MFCC cosine distance: {mfcc_results[c][s]:.4f}")

    return ttest_results, anova_results, mfcc_results

def compare_deltas_between_couples(data):
    print("\n--- Comparing normal-aroused deltas between couples ---")
    comparisons = []

    for i in range(len(couples)):
        for j in range(i+1, len(couples)):
            c1, c2 = couples[i], couples[j]
            for g in genders:
                delta_c1 = (data[c1][g]['aroused'].mean() - data[c1][g]['normal'].mean()).abs()
                delta_c2 = (data[c2][g]['aroused'].mean() - data[c2][g]['normal'].mean()).abs()
                diff = (delta_c1 - delta_c2).abs().sort_values(ascending=False)
                comparisons.append({
                    'couple1': c1,
                    'couple2': c2,
                    'gender': g,
                    'top_feature_differences': diff.head(5)
                })
    for comp in comparisons:
        print(f"\nTop feature differences between {comp['couple1']} and {comp['couple2']} for {comp['gender']}:")
        print(comp['top_feature_differences'])

    return comparisons




def run_grouped_feature_anova(data):
    print("\n--- Grouped feature ANOVA for normal vs aroused ---")
    grouped_results = {}

    feature_groups = get_feature_groups(next(iter(next(iter(next(iter(data.values())).values())).values())))

    for c in couples:
        grouped_results[c] = {}
        for g in genders:
            base_df = data[c][g]['normal']
            aroused_df = data[c][g]['aroused']
            grouped_results[c][g] = run_grouped_anova(base_df, aroused_df, feature_groups)
            print(f"Grouped ANOVA results for {c} {g}:")
            print(grouped_results[c][g])

    return grouped_results

    def anova(*dfs):


    if len(dfs) < 2:
        raise ValueError("Provide at least two DataFrames to compare.")
    
    shared_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        shared_cols &= set(df.columns)
    shared_cols = sorted(shared_cols)

    results = []

    for col in shared_cols:
        try:
            groups = [df[col].dropna() for df in dfs]
            f_stat, p_val = f_oneway(*groups)
            results.append({'feature': col, 'F': f_stat, 'p': p_val})
        except Exception as e:
            results.append({'feature': col, 'F': None, 'p': None})

    results_df = pd.DataFrame(results).sort_values('p')
    return results_df


def anova_per_group(dfs):
    """
    Run ANOVA per feature group across multiple DataFrames.

    Parameters:
        dfs: list of pandas DataFrames to compare (e.g., normal, aroused).

    Returns:
        dict: keys = group names, values = ANOVA results DataFrames sorted by p-value.
    """
    # Use the columns from the first df to get groups
    groups = get_feature_groups(dfs[0])

    results = {}
    for group_name, features in groups.items():
        # Filter only features present in all dfs
        common_features = [f for f in features if all(f in df.columns for df in dfs)]
        if not common_features:
            continue

        # Subset dfs by these features
        group_dfs = [df[common_features] for df in dfs]

        # Run ANOVA on this subset
        results[group_name] = anova(*group_dfs)

    return results

