import seaborn as sns
import matplotlib.pyplot as plt

def boxplots_for_top_N(significant_anova, significant_ttest,df):
    top_state_feats = significant_anova.sort_values('p-value').head(5)['feature'].tolist()
    top_sex_feats = significant_ttest.sort_values('p-value').head(5)['feature'].tolist()

    for feat in top_state_feats:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='state', y=feat, data=df)
        plt.title(f"{feat} by State")
        plt.tight_layout()
        plt.savefig(f"./output/graphs/boxplot_sex_{feat}.png")
        plt.close()

# Plot boxplots for sex
    for feat in top_sex_feats:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='sex', y=feat, data=df)
        plt.title(f"{feat} by Sex")
        plt.tight_layout()
        plt.savefig(f"./output/graphs/boxplot_sex_{feat}.png")
        plt.close()

def violinplots_for_top_N(significant_anova, significant_ttest, df):
    top_state_feats = significant_anova.sort_values('p-value').head(5)['feature'].tolist()
    top_sex_feats = significant_ttest.sort_values('p-value').head(5)['feature'].tolist()

    for feat in top_state_feats:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='state', y=feat, data=df)
        plt.title(f"{feat} by State (Violin Plot)")
        plt.tight_layout()
        plt.savefig(f"./output/graphs/violinplot_state_{feat}.png")
        plt.close()

    for feat in top_sex_feats:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='sex', y=feat, data=df)
        plt.title(f"{feat} by Sex (Violin Plot)")
        plt.tight_layout()
        plt.savefig(f"./output/graphs/violinplot_sex_{feat}.png")
        plt.close()
