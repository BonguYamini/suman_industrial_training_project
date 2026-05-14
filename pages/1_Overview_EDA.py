import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Overview & EDA", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("students.csv")
    # Clean dep column
    df['dep'] = df['dep'].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    df = df[df['dep'].isin(['Yes', 'No'])].copy()
    df['dep'] = df['dep'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

st.title("🏠 Overview & Exploratory Data Analysis")
st.markdown("---")

# Dataset snapshot
st.subheader("📋 Dataset Snapshot")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", df.shape[0])
col2.metric("Total Features", df.shape[1])
col3.metric("Depressed Students", int(df['dep'].sum()))
col4.metric("Non-Depressed", int((df['dep'] == 0).sum()))

st.markdown("---")

# Raw data
with st.expander("🔍 View Raw Data (first 20 rows)"):
    st.dataframe(df.head(20), use_container_width=True)

# Column info
st.subheader("📌 Column Information")
col_info = pd.DataFrame({
    "Column": df.columns,
    "Dtype": df.dtypes.values,
    "Non-Null Count": df.count().values,
    "Null Count": df.isnull().sum().values,
    "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
    "Unique Values": df.nunique().values
})
st.dataframe(col_info, use_container_width=True)

st.markdown("---")

# Missing values
st.subheader("🔴 Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(16, 4))
missing = df.isnull()
sns.heatmap(missing.T, cmap="YlOrRd", ax=ax, cbar=True, yticklabels=True)
ax.set_title("Missing Values per Column (Red = Missing)", fontsize=14)
ax.set_xlabel("Row Index")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# Statistical summary
st.subheader("📊 Statistical Summary (Numerical Columns)")
st.dataframe(df.describe().round(3), use_container_width=True)

st.markdown("---")

# Target distribution
st.subheader("🎯 Target Variable Distribution")
col1, col2 = st.columns(2)
with col1:
    dep_counts = df['dep'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['#2ecc71', '#e74c3c']
    ax.pie(dep_counts.values, labels=['No Depression', 'Depressed'],
           colors=colors, autopct='%1.1f%%', startangle=140,
           textprops={'fontsize': 13})
    ax.set_title("Depression Distribution", fontsize=15)
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### Key Observations")
    total = len(df)
    dep_pct = df['dep'].mean() * 100
    st.write(f"- **Total records after cleaning:** {total}")
    st.write(f"- **Depression prevalence:** {dep_pct:.1f}%")
    st.write(f"- **Dataset type:** Mixed (categorical + numerical)")
    st.write(f"- **Target column:** `dep` (0 = No, 1 = Yes)")
    st.write(f"- **Key features:** inter_dom, todep, tosc, toas, age, stay")
    st.write(f"- **Depression severity range (todep):** {df['todep'].min():.0f} – {df['todep'].max():.0f}")
    st.write(f"- **Mean acculturative stress (toas):** {df['toas'].mean():.1f}")

st.markdown("---")

# Key numeric feature distributions
st.subheader("📈 Key Numeric Feature Distributions")
num_features = ['todep', 'tosc', 'toas', 'age', 'stay', 'apd']
available = [c for c in num_features if c in df.columns]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(available):
    axes[i].hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].grid(axis='y', alpha=0.3)
plt.suptitle("Key Feature Distributions", fontsize=15, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# Categorical breakdown
st.subheader("🏷️ Categorical Variable Breakdown")
cat_cols = ['inter_dom', 'gender', 'academic', 'stay_cate', 'japanese_cate', 'english_cate', 'region']
cat_cols = [c for c in cat_cols if c in df.columns]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    vc = df[col].value_counts(dropna=True)
    axes[i].bar(vc.index, vc.values, color=plt.cm.Set2.colors[:len(vc)])
    axes[i].set_title(f'{col}', fontsize=11)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=30)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Categorical Variable Distributions", fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")
st.subheader("🔗 Correlation Matrix (Numerical Features)")
num_df = df.select_dtypes(include='number').dropna(axis=1, thresh=100)
fig, ax = plt.subplots(figsize=(14, 10))
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close()
