import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predict New Case", page_icon="🔮", layout="wide")

@st.cache_resource
def train_all_models():
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

    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "SVM":                 SVC(kernel='rbf', probability=True, random_state=42),
        "Naive Bayes":         GaussianNB(),
    }
    scaled_models = {"Logistic Regression", "SVM", "Naive Bayes"}

    trained = {}
    for name, model in models.items():
        if name in scaled_models:
            model.fit(X_train_sc, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model

    return trained, scaler, encoders, features, scaled_models

trained_models, scaler, encoders, features, scaled_models = train_all_models()

st.title("🔮 Predict New Student Case")
st.markdown("Fill in the student's details below and get a depression risk prediction from all 5 models.")
st.markdown("---")

st.subheader("📝 Student Information Form")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌐 Background**")
    inter_dom = st.selectbox("Student Type", ["Inter", "Dom"], help="International or Domestic")
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic = st.selectbox("Academic Level", ["Grad", "Under"])
    region = st.selectbox("Region", ["SEA", "EA", "SA", "JAP", "Others"])
    age = st.slider("Age", 18, 45, 22)
    stay = st.slider("Length of Stay (years)", 1, 10, 2)

with col2:
    st.markdown("**📚 Academic & Language**")
    japanese = st.slider("Japanese Proficiency (1=Low, 5=High)", 1, 5, 3)
    english = st.slider("English Proficiency (1=Low, 5=High)", 1, 5, 4)
    religion = st.selectbox("Religious Practice", ["Yes", "No"])
    suicide = st.selectbox("Suicidal Ideation", ["No", "Yes"])
    intimate = st.selectbox("Intimate Relationship", ["No", "Yes"])

with col3:
    st.markdown("**🧠 Mental Health Scores**")
    todep = st.slider("Total Depression Score (todep)", 0, 63, 15,
                      help="PHQ-9 based score. Higher = more depressed")
    tosc = st.slider("Social Connectedness (tosc)", 20, 48, 35,
                     help="Higher score = more connected")
    toas = st.slider("Acculturative Stress (toas)", 36, 144, 72,
                     help="Higher = more stressed")

st.markdown("---")
st.subheader("🔬 Stress Sub-domain Scores")
col4, col5, col6, col7 = st.columns(4)
with col4:
    apd = st.slider("Perceived Discrimination (apd)", 6, 36, 15)
with col5:
    ahome = st.slider("Homesickness (ahome)", 5, 20, 10)
with col6:
    aph = st.slider("Perceived Hatred (aph)", 3, 18, 7)
with col7:
    afear = st.slider("Fear (afear)", 3, 18, 6)

col8, col9, col10 = st.columns(3)
with col8:
    acs = st.slider("Culture Shock (acs)", 3, 18, 8)
with col9:
    aguilt = st.slider("Guilt (aguilt)", 3, 18, 5)
with col10:
    amiscell = st.slider("Miscellaneous Stress (amiscell)", 8, 48, 20)

st.markdown("---")

# Build input
input_dict = {
    'inter_dom': inter_dom,
    'gender': gender,
    'academic': academic,
    'age': age,
    'stay': stay,
    'japanese': japanese,
    'english': english,
    'religion': religion,
    'suicide': suicide,
    'todep': todep,
    'tosc': tosc,
    'toas': toas,
    'apd': apd,
    'ahome': ahome,
    'aph': aph,
    'afear': afear,
    'acs': acs,
    'aguilt': aguilt,
    'amiscell': amiscell,
}

input_row = {f: input_dict.get(f, 0) for f in features}
for col, le in encoders.items():
    if col in input_row:
        val = str(input_row[col])
        if val in le.classes_:
            input_row[col] = le.transform([val])[0]
        else:
            input_row[col] = 0

input_df = pd.DataFrame([input_row])
input_df_sc = scaler.transform(input_df)

if st.button("🚀 Predict Depression Risk", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    predictions = {}
    probabilities = {}
    model_colors = {
        "Logistic Regression": "#3498db",
        "Decision Tree": "#e67e22",
        "Random Forest": "#27ae60",
        "SVM": "#9b59b6",
        "Naive Bayes": "#e74c3c"
    }

    for name, model in trained_models.items():
        X_in = input_df_sc if name in scaled_models else input_df
        pred = model.predict(X_in)[0]
        prob = model.predict_proba(X_in)[0][1]
        predictions[name] = pred
        probabilities[name] = prob

    # Display prediction cards
    cols = st.columns(5)
    for i, (name, model) in enumerate(trained_models.items()):
        pred = predictions[name]
        prob = probabilities[name]
        with cols[i]:
            if pred == 1:
                st.error(f"**{name}**\n\n⚠️ **DEPRESSED**\n\n Prob: {prob*100:.1f}%")
            else:
                st.success(f"**{name}**\n\n✅ **NOT DEPRESSED**\n\nProb: {prob*100:.1f}%")

    st.markdown("---")

    # Ensemble / Voting
    dep_votes = sum(predictions.values())
    not_dep_votes = 5 - dep_votes
    ensemble = "DEPRESSION RISK ⚠️" if dep_votes >= 3 else "NO DEPRESSION ✅"
    avg_prob = np.mean(list(probabilities.values()))

    st.subheader("🗳️ Ensemble Voting Result")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Models Predicting Depression", f"{dep_votes}/5")
    col_b.metric("Models Predicting No Depression", f"{not_dep_votes}/5")
    col_c.metric("Average Probability of Depression", f"{avg_prob*100:.1f}%")

    if dep_votes >= 3:
        st.error(f"## 🚨 Ensemble Verdict: {ensemble}")
        st.warning("""
        **Recommendation:** This student shows indicators of depression.
        Consider referring to a counsellor or mental health professional.
        Key risk factors: high todep score, low social connectedness, high acculturative stress.
        """)
    else:
        st.success(f"## ✅ Ensemble Verdict: {ensemble}")
        st.info("The student appears to be in a healthy mental state based on the provided data. Continue monitoring well-being.")

    # Probability bar chart
    st.markdown("---")
    st.subheader("📊 Model-wise Depression Probability")
    import plotly.graph_objects as go
    model_names = list(probabilities.keys())
    probs = [probabilities[n]*100 for n in model_names]
    colors_bar = ['#e74c3c' if p >= 50 else '#2ecc71' for p in probs]
    fig = go.Figure(go.Bar(x=model_names, y=probs, marker_color=colors_bar,
                           text=[f"{p:.1f}%" for p in probs], textposition='auto'))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% threshold")
    fig.update_layout(title="Depression Probability by Model",
                      yaxis_title="Probability (%)", yaxis_range=[0, 100],
                      plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Input summary
    with st.expander("📋 View Input Summary"):
        summary = pd.DataFrame([input_dict]).T.reset_index()
        summary.columns = ['Feature', 'Value']
        st.dataframe(summary, use_container_width=True)

st.markdown("---")
st.markdown("""
**Legend:**
- `todep` → Total Depression Score (PHQ-9 based, 0-63)
- `tosc` → Social Connectedness Score (higher = more connected)
- `toas` → Total Acculturative Stress Score (higher = more stressed)
""")
