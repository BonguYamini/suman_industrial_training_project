# 🎓 Student Mental Health Analysis – Streamlit App

## Project Overview
A comprehensive machine learning web application analyzing student mental health data.
**Target:** Predict depression (`dep`) in international vs domestic students.
**Dataset:** 286 students, 50 features.

---

## 📁 Project Structure
```
student_project/
├── app.py                    ← Main landing page
├── students.csv              ← Dataset (copy your file here)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
└── pages/
    ├── 1_Overview_EDA.py     ← Page 1: Overview & EDA
    ├── 2_Visualisations.py   ← Page 2: 10 Chart Types
    ├── 3_ML_Models.py        ← Page 3: 5 ML Algorithms
    ├── 4_Model_Comparison.py ← Page 4: Model Comparison
    └── 5_Predict.py          ← Page 5: Predict New Case
```

---

## ⚙️ Setup Instructions (Step by Step)

### Step 1: Install Python (if not installed)
- Download Python 3.10+ from https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation

### Step 2: Place dataset
- Make sure `students.csv` is in the **same folder** as `app.py`
- (The file has been pre-converted from your .xls file)

### Step 3: Open Terminal / Command Prompt
- **Windows:** Press `Win + R` → type `cmd` → Enter
- **Mac/Linux:** Open Terminal

### Step 4: Navigate to project folder
```bash
cd path/to/student_project
# Example: cd C:\Users\YourName\Desktop\student_project
```

### Step 5: Install dependencies
```bash
pip install -r requirements.txt
```
*(Takes 1–2 minutes on first run)*

### Step 6: Run the app
```bash
streamlit run app.py
```

### Step 7: View in browser
- App opens automatically at: **http://localhost:8501**
- Use the **sidebar** to navigate between pages

---

## 🖥️ App Pages

| Page | Content |
|------|---------|
| 🏠 Overview & EDA | Dataset shape, missing values, distributions, correlation |
| 📊 Visualisations | 10 chart types: bar, pie, histogram, box, scatter, heatmap, violin, stacked bar, line, radar |
| 🤖 ML Models | Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes – with confusion matrices, ROC curves, feature importances |
| ⚖️ Model Comparison | Side-by-side metrics table, ROC comparison, radar chart, ranking |
| 🔮 Predict New Case | Interactive form → real-time prediction from all 5 models |

---

## 📊 ML Models Used
1. **Logistic Regression** – Linear classifier, baseline model
2. **Decision Tree** – Interpretable tree-based splits
3. **Random Forest** – Ensemble of 100 trees
4. **SVM** – RBF kernel support vector machine
5. **Naive Bayes** – Probabilistic Gaussian classifier

---

## 🎯 Target Variable
- `dep`: 0 = Not Depressed, 1 = Depressed
- Prediction is binary classification

---

## ⚡ Quick Troubleshooting
| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `students.csv not found` | Make sure CSV is in same folder as `app.py` |
| Port already in use | Run `streamlit run app.py --server.port 8502` |
| Slow loading | First load trains models; subsequent loads are cached |
