import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv("students.csv")
    df['dep'] = df['dep'].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    df = df[df['dep'].isin(['Yes', 'No'])].copy()
    df['dep'] = df['dep'].map({'Yes': 1, 'No': 0})

    features = ['inter_dom', 'gender', 'academic', 'age', 'stay', 'japanese', 'english',
                'religion', 'suicide', 'todep', 'tosc', 'toas', 'apd', 'ahome',
                'aph', 'afear', 'acs', 'aguilt', 'amiscell']
    features = [f for f in features if f in df.columns]

    X = df[features].copy()
    y = df['dep'].copy()
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=5, random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), False),
        "SVM":                 (SVC(kernel='rbf', probability=True, random_state=42), True),
        "Naive Bayes":         (GaussianNB(), True),
    }

    results = {}
    roc_data = {}
    cv_data = {}

    for name, (model, scaled) in models.items():
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc if scaled else X_test
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        cv_sc = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')

        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "AUC-ROC": roc_auc}
        roc_data[name] = (fpr, tpr, roc_auc)
        cv_data[name] = cv_sc

    return results, roc_data, cv_data, y_test

results, roc_data, cv_data, y_test = load_and_train()

st.title("⚖️ Model Comparison Dashboard")
st.markdown("Side-by-side performance analysis of all 5 ML models")
st.markdown("---")

# Summary table
res_df = pd.DataFrame(results).T.round(4)
res_df_pct = (res_df * 100).round(2)

st.subheader("📊 Performance Summary Table")
styled = res_df_pct.style\
    .background_gradient(cmap='RdYlGn', vmin=50, vmax=100)\
    .format("{:.2f}%")\
    .set_properties(**{'text-align': 'center'})
st.dataframe(styled, use_container_width=True)

# Best model
best_model = res_df['F1 Score'].idxmax()
best_f1 = res_df['F1 Score'].max()
best_acc = res_df['Accuracy'].max()
best_auc = res_df['AUC-ROC'].max()
st.success(f"🏆 **Best Model by F1 Score:** {best_model} — F1: {best_f1*100:.2f}% | AUC: {res_df.loc[best_model,'AUC-ROC']*100:.2f}%")

st.markdown("---")

# Grouped bar chart
st.subheader("📈 Metric Comparison – Grouped Bar Chart")
metrics_melted = res_df_pct.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
metrics_melted.rename(columns={'index': 'Model'}, inplace=True)
fig1 = px.bar(metrics_melted, x='Metric', y='Score', color='Model', barmode='group',
              color_discrete_sequence=px.colors.qualitative.Set1,
              title="All Models – All Metrics (%)", height=450,
              labels={'Score': 'Score (%)'})
fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ROC Curves
st.subheader("📉 ROC Curve Comparison")
colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
fig2, ax = plt.subplots(figsize=(9, 6))
for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
    ax.plot(fpr, tpr, color=colors_roc[i], lw=2, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – All Models', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# Cross-validation boxplot
st.subheader("🔄 5-Fold Cross Validation Scores")
cv_df_list = []
for name, scores in cv_data.items():
    for s in scores:
        cv_df_list.append({'Model': name, 'CV Accuracy': s * 100})
cv_df = pd.DataFrame(cv_df_list)

fig3 = px.box(cv_df, x='Model', y='CV Accuracy', color='Model',
              color_discrete_sequence=px.colors.qualitative.Set1,
              title="Cross-Validation Accuracy Distribution (5-Fold)",
              labels={'CV Accuracy': 'Accuracy (%)'}, height=420)
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Radar comparison
st.subheader("🕸️ Radar Chart – Model Profiles")
metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
fig4 = go.Figure()
color_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
for i, (model_name, row) in enumerate(res_df.iterrows()):
    vals = [row[m] * 100 for m in metrics_list]
    fig4.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=metrics_list + [metrics_list[0]],
        fill='toself',
        name=model_name,
        line_color=color_list[i],
        fillcolor=color_list[i].replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color_list[i] else color_list[i],
        opacity=0.7
    ))
fig4.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
    title="Model Performance Radar (all metrics in %)",
    height=550
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Accuracy heatmap
st.subheader("🌡️ Metric Heatmap")
fig5, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(res_df_pct, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=1, linecolor='white', ax=ax,
            cbar_kws={"label": "Score (%)"})
ax.set_title("Heatmap of All Model Metrics (%)", fontsize=14)
plt.tight_layout()
st.pyplot(fig5)
plt.close()

st.markdown("---")

# Ranking
st.subheader("🏅 Final Model Rankings")
ranking_df = res_df_pct.copy()
ranking_df['Overall Score'] = ranking_df.mean(axis=1)
ranking_df = ranking_df.sort_values('Overall Score', ascending=False)
ranking_df.insert(0, 'Rank', range(1, len(ranking_df)+1))
st.dataframe(ranking_df.style.format({c: '{:.2f}%' for c in ranking_df.columns if c != 'Rank'})
             .background_gradient(subset=['Overall Score'], cmap='Blues'),
             use_container_width=True)

st.info(f"🥇 **Winner: {ranking_df.index[0]}** with an overall average score of {ranking_df['Overall Score'].iloc[0]:.2f}%")
