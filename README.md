# Healthcare analysis & predictions

# 🏥 Healthcare ML Prediction System

> A multi-task machine learning pipeline that predicts **Test Results**, **Hospital Length of Stay**, and **Admission Type** from patient healthcare records — using classical ML algorithms with full EDA, feature engineering, and an end-to-end prediction function.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Prediction Tasks](#-prediction-tasks)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Quickstart](#-quickstart)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Models & Evaluation](#-models--evaluation)
- [Feature Importance](#-feature-importance)
- [Making Predictions](#-making-predictions)
- [Sample Predictions](#-sample-predictions)
- [Key Insights](#-key-insights)
- [Valid Input Values](#-valid-input-values)

---

## 🔍 Overview

This project applies supervised machine learning to a real-world healthcare dataset to solve three simultaneous prediction problems. The notebook is structured as a clean, step-by-step pipeline — from raw data ingestion through EDA, feature engineering, model training, evaluation, and a ready-to-use prediction function.

The system is designed to be easily extended and requires no external model serving — everything runs in a single Jupyter Notebook.

---

## 🎯 Prediction Tasks

| # | Task | Type | Target Column |
|---|------|------|---------------|
| 1 | **Test Result Prediction** | Multi-class Classification | `Test Results` → Normal / Abnormal / Inconclusive |
| 2 | **Length of Stay Prediction** | Regression | `Length_of_Stay` (days, derived) |
| 3 | **Admission Type Prediction** | Multi-class Classification | `Admission Type` → Emergency / Routine / Urgent |

---

## 📂 Dataset

**File:** `Healthcare Analysis Dataset.xlsx`

Place the dataset in the same directory as the notebook before running.

**Key columns used:**

| Column | Description |
|--------|-------------|
| `Age` | Patient age |
| `Gender` | Female / Male / Non-binary |
| `Blood Type` | ABO+Rh blood group |
| `Medical Condition` | Primary diagnosis |
| `Insurance Provider` | Patient's insurance plan |
| `Billing Amount` | Total hospital billing in USD |
| `Admission Type` | Emergency / Elective / Urgent |
| `Medication` | Primary medication prescribed |
| `Date of Admission` | Used to compute length of stay |
| `Discharge Date` | Used to compute length of stay |

> **Derived feature:** `Length_of_Stay = Discharge Date − Date of Admission` (days)

**Columns dropped to prevent leakage:** `Patient ID`, `Doctor`, `Hospital`, `Room Number`, `Date of Admission`, `Discharge Date`, `Hospital Latitude`, `Hospital Longitude`

---

## 📁 Project Structure

```
healthcare-ml-prediction/
│
├── Healthcare_ML_Analysis.ipynb       # Main notebook (all steps)
├── Healthcare Analysis Dataset.xlsx   # Source dataset (required)
└── README.md
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, wrangling, feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` + `seaborn` | EDA visualizations and model comparison charts |
| `scikit-learn` | ML models, preprocessing, metrics |

**Models used:**

- `LogisticRegression` — Task 1 (baseline classifier)
- `RandomForestClassifier` — Tasks 1 & 3 (best classifier)
- `LinearRegression` — Task 2 (baseline regressor)
- `RandomForestRegressor` — Task 2 (best regressor)
- `DecisionTreeClassifier` — Task 3 (interpretable baseline)

---

## ⚡ Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/healthcare-ml-prediction.git
cd healthcare-ml-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

### 3. Add the dataset

Place `Healthcare Analysis Dataset.xlsx` in the project root directory.

### 4. Launch the notebook

```bash
jupyter notebook Healthcare_ML_Analysis.ipynb
```

### 5. Run all cells

Use **Kernel → Restart & Run All** to execute the complete pipeline from top to bottom.

---

## 🔄 Pipeline Walkthrough

The notebook is organized into 7 sequential steps:

```
STEP 1 — Load & Understand Data
   ↓  Shape, dtypes, missing values, target distributions
STEP 2 — Data Preparation
   ↓  Date parsing, feature derivation, column dropping, null handling
STEP 3 — Prepare Data for Training
   ↓  Label encoding, feature/target split, train-test split (80/20), scaling
STEP 4 — Train Models
   ↓  6 models trained across 3 tasks
STEP 5 — Evaluate Models
   ↓  Accuracy, classification reports, MAE, RMSE, R²
STEP 6 — Insights & Feature Importance
   ↓  Random Forest feature importances, pattern visualizations
STEP 7 — Final Prediction System
   ↓  predict_patient() function — ready to use on new inputs
```

---

## 📊 Models & Evaluation

### Task 1 — Test Result Prediction (Classification)

| Model | Metric |
|-------|--------|
| Logistic Regression | Accuracy (see notebook output) |
| **Random Forest** ✅ | **Best accuracy** |

### Task 2 — Length of Stay Prediction (Regression)

| Model | Metrics |
|-------|---------|
| Linear Regression | MAE, RMSE, R² (see notebook output) |
| **Random Forest** ✅ | **Best MAE & R²** |

### Task 3 — Admission Type Prediction (Classification)

| Model | Metric |
|-------|--------|
| Decision Tree | Accuracy (see notebook output) |
| **Random Forest** ✅ | **Best accuracy** |

> Random Forest consistently outperforms baseline models across all three tasks. Exact metric values are printed live during notebook execution.

---

## 🌲 Feature Importance

Feature importances are extracted from all three Random Forest models and visualized as horizontal bar charts. The top contributing features are identified for each prediction task:

- **Test Result:** Driven by billing amount, age, and medical condition
- **Length of Stay:** Driven by billing amount, age, and medical condition
- **Admission Type:** Driven by billing amount, medical condition, and medication

> The notebook prints the top-3 most important features for each task with exact importance scores.

---

## 🔮 Making Predictions

Use the `predict_patient()` function defined in Step 7 to generate predictions for any new patient:

```python
result = predict_patient(
    age=45,
    gender='Female',
    blood_type='AB+',
    medical_condition='Cancer',
    insurance_provider='UnitedHealthCare',
    billing_amount=60000,
    admission_type='Emergency',
    medication='Lipitor'
)
```

**Output:**

```
=======================================================
  🏥 PATIENT PREDICTION RESULTS
=======================================================

  Patient Profile:
    Age                      : 45
    Gender                   : Female
    ...

  🔬 Test Result      : Abnormal
     Probabilities    : {'Abnormal': '72.00%', 'Inconclusive': '18.00%', 'Normal': '10.00%'}

  🛏  Length of Stay   : 14.3 days

  🚨 Admission Type   : Emergency
     Probabilities    : {'Elective': '8.00%', 'Emergency': '85.00%', 'Urgent': '7.00%'}
=======================================================
```

The function returns a dictionary with all three predictions:

```python
{
    'Test Result':    'Abnormal',
    'Length of Stay': 14.3,
    'Admission Type': 'Emergency'
}
```

---

## 🧪 Sample Predictions

Three example patients are demonstrated in the notebook:

| Patient | Age | Condition | Billing | Predicted Stay |
|---------|-----|-----------|---------|----------------|
| Elderly diabetic male | 72 | Diabetes | $50,000 | See output |
| Young asthma patient | 28 | Asthma | $12,000 | See output |
| Middle-aged hypertension | 55 | Diabetes | $30,500 | See output |

---

## 💡 Key Insights

**What affects Test Results?**
- Billing amount, age, and medical condition are the dominant drivers
- Test result distributions are fairly balanced across medical conditions

**What increases Length of Stay?**
- Billing amount strongly correlates with longer stays
- Certain conditions (e.g., Cancer) tend to result in longer hospitalizations
- Emergency admissions generally have longer stays than Elective

**What leads to Emergency Admission?**
- Medical condition type is the strongest predictor
- Age plays a role — older patients skew toward emergency admissions
- Higher billing amounts are associated with emergency cases

---

## ✅ Valid Input Values

When calling `predict_patient()`, use only the following values:

| Parameter | Valid Options |
|-----------|--------------|
| `gender` | `Female`, `Male`, `Non-binary` |
| `blood_type` | `A+`, `A-`, `B+`, `B-`, `O+`, `O-`, `AB+`, `AB-` |
| `medical_condition` | `Hypertension`, `Cancer`, `Asthma`, `Diabetes`, `Obesity`, `Arthritis` |
| `insurance_provider` | `Medicare`, `UnitedHealthCare`, `Aetna`, `Cigna` |
| `admission_type` | `Emergency`, `Elective`, `Urgent` |
| `medication` | `Ibuprofen`, `Lipitor`, `Penicillin`, `Paracetamol`, `Aspirin` |

> Values outside these categories will raise a `LabelEncoder` error, as the encoders were fitted only on values present in the training data.

---

## 📄 License

This project is for educational and research purposes.
