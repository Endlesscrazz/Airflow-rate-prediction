# analyze_ground_truth.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load CSV ===
file_path = "airflow_ground_truth_gypsum_8_hole.csv"  # update path if needed
df = pd.read_csv(file_path)

# === Basic Overview ===
print("===== Basic Overview =====")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

print("===== Null Value Counts =====")
print(df.isnull().sum(), "\n")

# === Unique Airflow Rates ===
print("===== Airflow Rate Summary =====")
if "airflow_rate" in df.columns:
    unique_rates = df["airflow_rate"].value_counts().sort_index()
    print(f"Number of unique airflow rates: {df['airflow_rate'].nunique()}")
    print(unique_rates, "\n")
else:
    print("Column 'airflow_rate' not found!\n")

# === Visualization Setup ===
sns.set(style="whitegrid", palette="viridis")

# --- 1. Distribution of Airflow Rates ---
plt.figure(figsize=(12, 6))
sns.barplot(x=unique_rates.index, y=unique_rates.values, color='skyblue')
plt.xticks(rotation=75)
plt.xlabel("Airflow Rate")
plt.ylabel("Number of Videos")
plt.title("Number of Videos per Airflow Rate")
plt.tight_layout()
plt.show()

# --- 2. Missing Value Heatmap ---
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="coolwarm")
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.show()

# --- 3. Delta_T vs Airflow Rate Scatter Plot ---
if "delta_T" in df.columns and "airflow_rate" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="airflow_rate", y="delta_T", hue="session", palette="tab10")
    plt.title("ΔT vs Airflow Rate (colored by session)")
    plt.tight_layout()
    plt.show()

# --- 4. Grouped by Session (Optional) ---
if "session" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="session", order=sorted(df["session"].unique()))
    plt.title("Number of Videos per Session")
    plt.tight_layout()
    plt.show()


# python scripts/analyze_ground_truth.py