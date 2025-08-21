# Model 2: Lognormal Monte Carlo Simulation (Universal Loader with Silent Tkinter)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import io, os, sys, warnings
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# -----------------------------
# File Loader: Colab or Tkinter
# -----------------------------
def load_tabular_file():
    # 1) Try Google Colab
    try:
        import google.colab  # type: ignore
        from google.colab import files  # type: ignore
        print("Colab detected. Please upload your Excel/CSV file…")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        fname = list(uploaded.keys())[0]
        byts = io.BytesIO(uploaded[fname])
        if fname.lower().endswith(".csv"):
            return pd.read_csv(byts), fname
        else:
            try:
                return pd.read_excel(byts, sheet_name=0, header=1), fname
            except Exception:
                return pd.read_excel(byts, sheet_name=0, header=0), fname
    except Exception:
        pass

    # 2) Local: Tkinter file picker (silence Cocoa warning on macOS)
    print("Please select your Excel/CSV file from the dialog…")
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')  # suppress macOS NSOpenPanel warning
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Excel or CSV file",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    sys.stderr = stderr  # restore stderr

    if not path:
        raise FileNotFoundError("No file selected.")
    if path.lower().endswith(".csv"):
        return pd.read_csv(path), os.path.basename(path)
    else:
        try:
            return pd.read_excel(path, sheet_name=0, header=1), os.path.basename(path)
        except Exception:
            return pd.read_excel(path, sheet_name=0, header=0), os.path.basename(path)

# -----------------------------
# Load Data
# -----------------------------
df, source_name = load_tabular_file()
print(f"Loaded file: {source_name}")

# Column normalization
try:
    df.rename(columns={
        df.columns[2]: 'WBS Code',
        df.columns[3]: 'Task',
        df.columns[5]: 'Cost'
    }, inplace=True)
except Exception:
    pass

lower_cols = {str(c).strip().lower(): c for c in df.columns}
if 'wbs code' in lower_cols and 'WBS Code' not in df.columns:
    df.rename(columns={lower_cols['wbs code']: 'WBS Code'}, inplace=True)
if 'task' in lower_cols and 'Task' not in df.columns:
    df.rename(columns={lower_cols['task']: 'Task'}, inplace=True)
for key in ('cost', 'most likely', 'most likely cost'):
    if key in lower_cols and 'Cost' not in df.columns:
        df.rename(columns={lower_cols[key]: 'Cost'}, inplace=True)
        break

df_clean = df[pd.to_numeric(df.get('Cost', pd.Series(index=df.index)), errors='coerce').notna()].copy()
df_clean['Cost'] = pd.to_numeric(df_clean['Cost'])

# Unique labels
if 'WBS Code' in df_clean.columns and 'Task' in df_clean.columns:
    df_clean['Unique Task'] = df_clean['WBS Code'].astype(str) + " - " + df_clean['Task'].astype(str)
    task_labels = df_clean['Unique Task'].tolist()
else:
    task_labels = df_clean.get('Task', pd.Series([f"Task {i}" for i in range(len(df_clean))])).astype(str).tolist()

# -----------------------------
# Parameters for Lognormal
# -----------------------------
df_clean['StdDev'] = df_clean['Cost'] * 0.2  # assume 20% stdev
df_clean['Mu'] = np.log(df_clean['Cost']**2 / np.sqrt(df_clean['StdDev']**2 + df_clean['Cost']**2))
df_clean['Sigma'] = np.sqrt(np.log(1 + (df_clean['StdDev']**2 / df_clean['Cost']**2)))

# -----------------------------
# Monte Carlo simulation (Lognormal)
# -----------------------------
NUM_SIMULATIONS = 10000
task_sim_matrix = np.zeros((NUM_SIMULATIONS, len(df_clean)))

for i, row in df_clean.iterrows():
    task_sim_matrix[:, i] = np.random.lognormal(
        mean=row['Mu'], sigma=row['Sigma'], size=NUM_SIMULATIONS
    )

total_costs = task_sim_matrix.sum(axis=1)

# -----------------------------
# Summary
# -----------------------------
mean_cost = float(total_costs.mean())
median_cost = float(np.percentile(total_costs, 50))
p90_cost = float(np.percentile(total_costs, 90))
p10_cost = float(np.percentile(total_costs, 10))
std_dev   = float(total_costs.std())

print("\n=== Lognormal Model Summary ===")
print(f"Mean Cost: £{mean_cost:,.2f}")
print(f"P50 (Median): £{median_cost:,.2f}")
print(f"P90: £{p90_cost:,.2f}")
print(f"P10: £{p10_cost:,.2f}")
print(f"Standard Deviation: £{std_dev:,.2f}")

# -----------------------------
# Plots
# -----------------------------
# Distribution
plt.figure(figsize=(10, 6))
sns.histplot(total_costs, bins=60, kde=True, color='salmon')
plt.axvline(mean_cost, color='red', linestyle='--', label=f'Mean: £{mean_cost:,.0f}')
plt.axvline(p90_cost, color='orange', linestyle='--', label=f'P90: £{p90_cost:,.0f}')
plt.axvline(p10_cost, color='purple', linestyle='--', label=f'P10: £{p10_cost:,.0f}')
plt.title('Model 2: Lognormal Simulation of Total Project Cost')
plt.xlabel('Total Project Cost (£)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Sensitivity
sensitivity = []
for i, label in enumerate(task_labels):
    without = total_costs - task_sim_matrix[:, i]
    delta = total_costs.mean() - without.mean()
    sensitivity.append((label, float(delta)))
sensitivity_df = pd.DataFrame(sensitivity, columns=['Task', 'Mean Contribution']).sort_values('Mean Contribution')

plt.figure(figsize=(10, 6))
sns.barplot(x='Mean Contribution', y='Task', data=sensitivity_df, palette='Reds_r')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Tornado Chart: Sensitivity of Tasks (Lognormal)")
plt.xlabel("Mean Contribution (£)")
plt.ylabel("Task")
plt.tight_layout()
plt.show()

# Correlation
correlations = []
for i, label in enumerate(task_labels):
    corr, _ = pearsonr(task_sim_matrix[:, i], total_costs)
    correlations.append((label, float(corr)))
correlation_df = pd.DataFrame(correlations, columns=['Task', 'Correlation Coefficient']).sort_values('Correlation Coefficient')

plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation Coefficient', y='Task', data=correlation_df, palette='coolwarm')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Correlation: Task Cost vs Total Cost (Lognormal)")
plt.xlabel("Pearson Correlation")
plt.ylabel("Task")
plt.tight_layout()
plt.show()