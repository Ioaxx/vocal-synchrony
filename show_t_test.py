import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your results
df = pd.read_csv("female_ttest_results.csv")

# Add significance column
df['significant'] = df['p_value'] < 0.05

# Sort by absolute t-stat for better visual order
df = df.sort_values(by='t_stat', key=abs, ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='feature', y='t_stat', hue='significant', palette={True: 'tomato', False: 'gray'})

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("T-Statistics per Feature (Female aroused vs baseline)")
plt.ylabel("T-Statistic")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.legend(title="p < 0.05")
plt.tight_layout()
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "female_ttest_plot.png"))
plt.close()


# Load your results
df = pd.read_csv("male_ttest_results.csv")

# Add significance column
df['significant'] = df['p_value'] < 0.05

# Sort by absolute t-stat for better visual order
df = df.sort_values(by='t_stat', key=abs, ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='feature', y='t_stat', hue='significant', palette={True: 'tomato', False: 'gray'})

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("T-Statistics per Feature (Male aroused vs baseline)")
plt.ylabel("T-Statistic")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.legend(title="p < 0.05")
plt.tight_layout()
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Male_ttest_plot.png"))
plt.close()
