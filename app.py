import streamlit as st

st.set_page_config(
    page_title="Student Mental Health Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f0f4ff;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Student Mental Health Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A comprehensive machine learning study on international vs domestic students</div>', unsafe_allow_html=True)

st.sidebar.title("📚 Navigation")
st.sidebar.markdown("---")

pages = {
    "🏠 Overview & EDA": "pages/1_Overview_EDA.py",
    "📊 Data Visualisations": "pages/2_Visualisations.py",
    "🤖 ML Models": "pages/3_ML_Models.py",
    "⚖️ Model Comparison": "pages/4_Model_Comparison.py",
    "🔮 Predict New Case": "pages/5_Predict.py",
}

st.sidebar.markdown("### Pages")
for page_name in pages:
    st.sidebar.markdown(f"• {page_name}")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Page 1** → Overview & EDA\nExplore dataset statistics, missing values, and distributions.")
with col2:
    st.info("**Page 2** → Visualisations\n10 different chart types to understand the data deeply.")
with col3:
    st.info("**Page 3** → ML Models\n5 algorithms trained and evaluated with metrics.")

col4, col5 = st.columns(2)
with col4:
    st.success("**Page 4** → Model Comparison\nSide-by-side performance comparison of all models.")
with col5:
    st.success("**Page 5** → Predict New Case\nEnter student details and get a depression prediction.")

st.markdown("---")
st.markdown("**Dataset:** Students Mental Health | **Target:** Depression (dep) | **Records:** 286 | **Features:** 50")
