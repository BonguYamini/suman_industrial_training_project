import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score, precision_score,
                              recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Models", page_icon="🤖", layout="wide")

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("students.csv")
    df['dep'] = df['dep'].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    df = df[df['dep'].isin(['Yes', 'No'])].copy()
    df['dep'] = df['dep'].map({'Yes': 1, 'No': 0})

    features = ['inter_dom', 'gender', 'academic', 'age', 'stay', 'japanese', 'english',
                'religion', 'suicide', 'todep', 'tosc', 'toas', 'apd', 'ahome',
                'aph', 'afear', 'acs', 'aguilt', 'amiscell']
    features = [f for f in features if f in df.columns]
    target = 'dep'

    X = df[features].copy()
    y = df[target].copy()

    # Encode categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])

    # Fill missing with median
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X, y, X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, features

X, y, X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, features = load_and_prepare()

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Dep', 'Dep'],
                yticklabels=['No Dep', 'Dep'],
                linewidths=1)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    return fig

def plot_roc(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve – {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, roc_auc

def show_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Precision", f"{prec*100:.2f}%")
    col3.metric("Recall", f"{rec*100:.2f}%")
    col4.metric("F1 Score", f"{f1*100:.2f}%")
    return acc, prec, rec, f1

st.title("🤖 Machine Learning Models")
st.markdown("5 classification algorithms trained on student mental health data")
st.markdown(f"**Train set:** {len(X_train)} samples | **Test set:** {len(X_test)} samples | **Features used:** {len(features)}")
st.markdown("---")

# ─── 1. Logistic Regression ──────────────────────────────────────────────────
st.subheader("1️⃣ Logistic Regression")
with st.spinner("Training Logistic Regression..."):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]

lr_acc, lr_prec, lr_rec, lr_f1 = show_metrics(y_test, lr_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, lr_pred), "LR – Confusion Matrix"))
with col2:
    fig_roc, lr_auc = plot_roc(y_test, lr_prob, "Logistic Regression")
    st.pyplot(fig_roc)
with col3:
    # Feature importance (coefficients)
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lr.coef_[0]})
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index).head(10)
    fig_c, ax = plt.subplots(figsize=(5, 4))
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    ax.set_title("Top Feature Coefficients")
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_c)

with st.expander("📋 Full Classification Report – Logistic Regression"):
    st.text(classification_report(y_test, lr_pred, target_names=['No Depression', 'Depression']))

st.markdown("**Cross-validation accuracy:** " + f"{cross_val_score(lr, X_train_sc, y_train, cv=5).mean()*100:.2f}%")
st.markdown("---")

# ─── 2. Decision Tree ────────────────────────────────────────────────────────
st.subheader("2️⃣ Decision Tree")
with st.spinner("Training Decision Tree..."):
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]

dt_acc, dt_prec, dt_rec, dt_f1 = show_metrics(y_test, dt_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, dt_pred), "DT – Confusion Matrix"))
with col2:
    fig_roc, dt_auc = plot_roc(y_test, dt_prob, "Decision Tree")
    st.pyplot(fig_roc)
with col3:
    fig_imp, ax = plt.subplots(figsize=(5, 4))
    imp_df = pd.DataFrame({'Feature': features, 'Importance': dt.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='#3498db')
    ax.set_title("Top Feature Importances")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_imp)

with st.expander("🌳 Decision Tree Diagram (first 3 levels)"):
    fig_tree, ax = plt.subplots(figsize=(18, 6))
    plot_tree(dt, feature_names=features, class_names=['No Dep', 'Dep'],
              filled=True, rounded=True, max_depth=3, ax=ax, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_tree)

with st.expander("📋 Full Classification Report – Decision Tree"):
    st.text(classification_report(y_test, dt_pred, target_names=['No Depression', 'Depression']))

st.markdown("**Cross-validation accuracy:** " + f"{cross_val_score(dt, X_train, y_train, cv=5).mean()*100:.2f}%")
st.markdown("---")

# ─── 3. Random Forest ────────────────────────────────────────────────────────
st.subheader("3️⃣ Random Forest")
with st.spinner("Training Random Forest..."):
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

rf_acc, rf_prec, rf_rec, rf_f1 = show_metrics(y_test, rf_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, rf_pred), "RF – Confusion Matrix"))
with col2:
    fig_roc, rf_auc = plot_roc(y_test, rf_prob, "Random Forest")
    st.pyplot(fig_roc)
with col3:
    fig_imp, ax = plt.subplots(figsize=(5, 4))
    imp_df = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
    ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1], color='#27ae60')
    ax.set_title("Top Feature Importances (RF)")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_imp)

with st.expander("📋 Full Classification Report – Random Forest"):
    st.text(classification_report(y_test, rf_pred, target_names=['No Depression', 'Depression']))

st.markdown("**Cross-validation accuracy:** " + f"{cross_val_score(rf, X_train, y_train, cv=5).mean()*100:.2f}%")
st.markdown("---")

# ─── 4. SVM ──────────────────────────────────────────────────────────────────
st.subheader("4️⃣ Support Vector Machine (SVM)")
with st.spinner("Training SVM..."):
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)
    svm_pred = svm.predict(X_test_sc)
    svm_prob = svm.predict_proba(X_test_sc)[:, 1]

svm_acc, svm_prec, svm_rec, svm_f1 = show_metrics(y_test, svm_pred)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, svm_pred), "SVM – Confusion Matrix"))
with col2:
    fig_roc, svm_auc = plot_roc(y_test, svm_prob, "SVM (RBF Kernel)")
    st.pyplot(fig_roc)

with st.expander("📋 Full Classification Report – SVM"):
    st.text(classification_report(y_test, svm_pred, target_names=['No Depression', 'Depression']))

st.markdown("**Cross-validation accuracy:** " + f"{cross_val_score(svm, X_train_sc, y_train, cv=5).mean()*100:.2f}%")
st.markdown("---")

# ─── 5. Naive Bayes ──────────────────────────────────────────────────────────
st.subheader("5️⃣ Naive Bayes (Gaussian)")
with st.spinner("Training Naive Bayes..."):
    nb = GaussianNB()
    nb.fit(X_train_sc, y_train)
    nb_pred = nb.predict(X_test_sc)
    nb_prob = nb.predict_proba(X_test_sc)[:, 1]

nb_acc, nb_prec, nb_rec, nb_f1 = show_metrics(y_test, nb_pred)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_confusion_matrix(confusion_matrix(y_test, nb_pred), "NB – Confusion Matrix"))
with col2:
    fig_roc, nb_auc = plot_roc(y_test, nb_prob, "Naive Bayes")
    st.pyplot(fig_roc)

with st.expander("📋 Full Classification Report – Naive Bayes"):
    st.text(classification_report(y_test, nb_pred, target_names=['No Depression', 'Depression']))

st.markdown("**Cross-validation accuracy:** " + f"{cross_val_score(nb, X_train_sc, y_train, cv=5).mean()*100:.2f}%")
st.markdown("---")

# Store results for comparison page
import json
results = {
    "Logistic Regression": {"Accuracy": lr_acc, "Precision": lr_prec, "Recall": lr_rec, "F1": lr_f1, "AUC": lr_auc},
    "Decision Tree":       {"Accuracy": dt_acc, "Precision": dt_prec, "Recall": dt_rec, "F1": dt_f1, "AUC": dt_auc},
    "Random Forest":       {"Accuracy": rf_acc, "Precision": rf_prec, "Recall": rf_rec, "F1": rf_f1, "AUC": rf_auc},
    "SVM":                 {"Accuracy": svm_acc, "Precision": svm_prec, "Recall": svm_rec, "F1": svm_f1, "AUC": svm_auc},
    "Naive Bayes":         {"Accuracy": nb_acc, "Precision": nb_prec, "Recall": nb_rec, "F1": nb_f1, "AUC": nb_auc},
}
with open("model_results.json", "w") as f:
    json.dump(results, f)

st.success("✅ All 5 models trained! Results saved. Navigate to Model Comparison page.")
