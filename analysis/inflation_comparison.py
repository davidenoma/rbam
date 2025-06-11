import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "Cohort": ["Crohn's disease", "Coronary arterial disease", "Hypertension",
               "Bipolar disorder", "Schizophrenia"],
    "REGENIE": [3.455, 3.246, 3.263, 3.608, 4.718],
    "SKAT": [7.747, 6.882, 7.512, 11.234, 112.766],
    "MOKA-ED": [4.247, 3.981, 4.192, 4.921, 10.085],
    "MOKA-ED-COR": [2.452, 2.513, 2.527, 2.623, 4.627],
}

df = pd.DataFrame(data).set_index("Cohort")

# Transpose so that methods become the x-axis
df_t = df.T   # Methods are now the index

# Plot with reduced spacing (larger bar width + slightly narrower figure)
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
df_t.plot(kind='bar', ax=ax, width=0.9)   # width close to 1 shrinks gaps

ax.set_ylabel("Genomic inflation factor (λGC)")
ax.set_xlabel("Method")
ax.set_title("Genomic inflation factor comparison (compact spacing)")

# Legend inside the plot
ax.legend(title="Cohort", loc="upper right", bbox_to_anchor=(0.98, 0.98))

ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("genomic_inflation_factor_comparison_compact.png", dpi=600)
plt.show()
