import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Visualisations", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("students.csv")
    df['dep'] = df['dep'].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    df = df[df['dep'].isin(['Yes', 'No'])].copy()
    df['dep_label'] = df['dep']
    df['dep'] = df['dep'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

st.title("📊 Data Visualisations")
st.markdown("10 comprehensive chart types for deep data exploration")
st.markdown("---")

# 1. Bar Chart - Depression by Student Type
st.subheader("1️⃣ Bar Chart – Depression by Student Type (International vs Domestic)")
fig = px.bar(
    df.groupby(['inter_dom', 'dep_label']).size().reset_index(name='count'),
    x='inter_dom', y='count', color='dep_label',
    barmode='group', color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
    labels={'inter_dom': 'Student Type', 'count': 'Number of Students', 'dep_label': 'Depressed'},
    title="Depression Count by Student Type"
)
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
st.plotly_chart(fig, use_container_width=True)
st.markdown("**Insight:** International students show a notably higher rate of depression compared to domestic students.")

st.markdown("---")

# 2. Pie / Donut Chart - Gender Distribution
st.subheader("2️⃣ Donut Chart – Depression by Gender")
col1, col2 = st.columns(2)
with col1:
    gender_dep = df.groupby(['gender', 'dep_label']).size().reset_index(name='count')
    fig2 = px.sunburst(gender_dep, path=['gender', 'dep_label'], values='count',
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       title="Gender → Depression (Sunburst)")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    fig2b, ax = plt.subplots(figsize=(5, 5))
    female_dep = df[df['gender']=='Female']['dep'].mean()*100
    male_dep = df[df['gender']=='Male']['dep'].mean()*100
    ax.bar(['Female', 'Male'], [female_dep, male_dep], color=['#e91e8c', '#3498db'], width=0.5)
    ax.set_ylabel('Depression Rate (%)')
    ax.set_title('Depression Rate by Gender (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate([female_dep, male_dep]):
        ax.text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig2b)
    plt.close()
st.markdown("**Insight:** Female students show a slightly higher depression rate than male students.")

st.markdown("---")

# 3. Histogram - Depression Score Distribution
st.subheader("3️⃣ Histogram – Depression Score (todep) by Depression Status")
fig3 = px.histogram(df, x='todep', color='dep_label', nbins=25, barmode='overlay',
                    color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                    labels={'todep': 'Total Depression Score', 'dep_label': 'Depressed'},
                    title="Distribution of Depression Scores")
fig3.update_traces(opacity=0.75)
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("**Insight:** Depressed students have significantly higher `todep` scores, confirming the score is a strong indicator.")

st.markdown("---")

# 4. Box Plot - Acculturative Stress by Student Type
st.subheader("4️⃣ Box Plot – Acculturative Stress (toas) by Student Type & Depression")
fig4 = px.box(df, x='inter_dom', y='toas', color='dep_label',
              color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
              labels={'inter_dom': 'Student Type', 'toas': 'Acculturative Stress Score', 'dep_label': 'Depressed'},
              title="Acculturative Stress by Student Type and Depression Status")
fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400)
st.plotly_chart(fig4, use_container_width=True)
st.markdown("**Insight:** International depressed students have the highest acculturative stress, showing its strong link to mental health.")

st.markdown("---")

# 5. Scatter Plot - Social Connectedness vs Depression Score
st.subheader("5️⃣ Scatter Plot – Social Connectedness (tosc) vs Depression Score (todep)")
fig5 = px.scatter(df.dropna(subset=['tosc', 'todep', 'inter_dom']),
                  x='tosc', y='todep', color='dep_label',
                  size='toas', hover_data=['gender', 'age', 'stay'],
                  color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                  labels={'tosc': 'Social Connectedness Score', 'todep': 'Depression Score'},
                  title="Social Connectedness vs Depression Score (bubble size = stress)")
fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=450)
st.plotly_chart(fig5, use_container_width=True)
st.markdown("**Insight:** Negative correlation — students with lower social connectedness tend to have higher depression scores.")

st.markdown("---")

# 6. Heatmap - Correlation
st.subheader("6️⃣ Heatmap – Feature Correlation with Depression")
num_cols = ['todep', 'tosc', 'toas', 'apd', 'ahome', 'aph', 'afear', 'acs', 'aguilt', 'amiscell', 'age', 'stay', 'dep']
corr_df = df[num_cols].corr()[['dep']].drop('dep').sort_values('dep', ascending=False)
fig6, ax = plt.subplots(figsize=(4, 8))
sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0,
            linewidths=1, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation with Depression\n(dep)", fontsize=13)
plt.tight_layout()
st.pyplot(fig6)
plt.close()
st.markdown("**Insight:** `todep` has the highest positive correlation. `tosc` (social connectedness) is negatively correlated.")

st.markdown("---")

# 7. Violin Plot - Stay Duration vs Depression Score
st.subheader("7️⃣ Violin Plot – Length of Stay vs Depression Score")
fig7 = px.violin(df.dropna(subset=['stay_cate', 'todep']),
                 x='stay_cate', y='todep', color='dep_label', box=True,
                 color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                 category_orders={'stay_cate': ['Short', 'Medium', 'Long']},
                 labels={'stay_cate': 'Stay Duration', 'todep': 'Depression Score'},
                 title="Depression Score by Length of Stay")
fig7.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=450)
st.plotly_chart(fig7, use_container_width=True)
st.markdown("**Insight:** Short-stay international students tend to show higher depression scores.")

st.markdown("---")

# 8. Stacked Bar - Depression by Academic Level
st.subheader("8️⃣ Stacked Bar Chart – Depression by Academic Level & Region")
acad_dep = df.groupby(['academic', 'dep_label']).size().unstack(fill_value=0)
fig8, ax = plt.subplots(figsize=(7, 5))
acad_dep.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title("Depression Count by Academic Level", fontsize=14)
ax.set_xlabel("Academic Level")
ax.set_ylabel("Count")
ax.legend(title="Depressed", labels=['No', 'Yes'])
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
st.pyplot(fig8)
plt.close()
st.markdown("**Insight:** Graduate students represent the majority of cases in this dataset.")

st.markdown("---")

# 9. Line Chart - Depression Score by Age
st.subheader("9️⃣ Line Chart – Average Depression Score by Age")
age_dep = df.dropna(subset=['age', 'todep']).groupby('age')['todep'].mean().reset_index()
fig9 = px.line(age_dep, x='age', y='todep', markers=True,
               labels={'age': 'Age', 'todep': 'Avg Depression Score'},
               title="Average Depression Score by Student Age")
fig9.update_traces(line_color='#9b59b6', line_width=3)
fig9.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400)
st.plotly_chart(fig9, use_container_width=True)
st.markdown("**Insight:** Younger students (18–22) tend to report higher average depression scores.")

st.markdown("---")

# 10. Radar Chart - Stress Sub-domains for Depressed vs Non-Depressed
st.subheader("🔟 Radar Chart – Stress Sub-domains: Depressed vs Non-Depressed")
stress_cols = ['apd', 'ahome', 'aph', 'afear', 'acs', 'aguilt', 'amiscell']
stress_labels = ['Perceived Discrimination', 'Homesickness', 'Perceived Hatred',
                 'Fear', 'Culture Shock', 'Guilt', 'Miscellaneous']
dep_means = df[df['dep']==1][stress_cols].mean().values
nodep_means = df[df['dep']==0][stress_cols].mean().values

fig10 = go.Figure()
fig10.add_trace(go.Scatterpolar(r=dep_means.tolist() + [dep_means[0]],
                                theta=stress_labels + [stress_labels[0]],
                                fill='toself', name='Depressed',
                                line_color='#e74c3c', fillcolor='rgba(231,76,60,0.2)'))
fig10.add_trace(go.Scatterpolar(r=nodep_means.tolist() + [nodep_means[0]],
                                theta=stress_labels + [stress_labels[0]],
                                fill='toself', name='Not Depressed',
                                line_color='#2ecc71', fillcolor='rgba(46,204,113,0.2)'))
fig10.update_layout(polar=dict(radialaxis=dict(visible=True)),
                    title="Acculturative Stress Sub-domains Comparison",
                    showlegend=True, height=500)
st.plotly_chart(fig10, use_container_width=True)
st.markdown("**Insight:** Depressed students score higher across ALL stress sub-domains, especially Perceived Discrimination and Homesickness.")

st.markdown("---")
st.success("✅ All 10 visualisation types completed! Navigate to ML Models for model training.")
