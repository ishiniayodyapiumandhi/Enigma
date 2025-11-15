# 🧠 ModelX Optimization Sprint - Dementia Risk Prediction

## 🎯 Project Overview
This project is part of the **ModelX Optimization Sprint Hackathon** focused on building a binary classification model to predict dementia risk using **non-medical features only**. The model helps normal people estimate their dementia risk using information they already know about themselves, without requiring medical tests or clinical assessments.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Team](#team)
- [License](#license)

## 💡 Business Problem

### Problem Statement
Dementia is a major and growing global health issue affecting millions worldwide. Many risk factors are non-medical, such as lifestyle, education, and social context. This project explores how well non-medical information alone can help predict dementia risk.

### Solution Approach
Build a machine learning model that estimates dementia risk probability (0-100%) and classifies individuals as "At Risk" or "Not At Risk" using only non-medical variables that people typically know about themselves.

### Use Case
Imagine a simple website or mobile app where users answer questions like:
- How old are you? 🎂
- What's your education level? 🎓
- Who do you live with? 🏠
- Do you smoke or drink alcohol? 🚬🍷
- Have you had a heart attack or stroke? ❤️‍🩹

The system would then provide:
- "Your estimated risk of having dementia is X%"
- "At Risk" or "Not At Risk" classification

## 📊 Dataset

### Source
**NACC Uniform Data Set (UDS) Version 3.0**
- Curated subset of the NACC cohort
- Each row represents one participant visit
- Contains both medical and non-medical features
- Binary label indicating dementia vs. no dementia

### Feature Constraints
**✅ Allowed Features (Non-Medical):**
- Demographic: Age, Gender, Education, Marital Status
- Lifestyle: Smoking, Alcohol, Physical Activity
- Social: Living Situation, Social Engagement
- Known Conditions: Heart attack, Stroke (if patient-aware)

**❌ Prohibited Features (Medical):**
- Cognitive test scores (MMSE, MoCA)
- Lab results and clinical measurements
- Brain scan results
- Specialist clinical assessments

## 🏗️ Project Structure

```
ModelX-Dementia-Risk-Prediction/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 environment.yml                    # Conda environment
│
├── 📁 data/
│   ├── 📄 raw/                          # Original dataset
│   ├── 📄 processed/                    # Cleaned & processed data
│   └── 📄 external/                     # External data sources
│
├── 📁 notebooks/
│   ├── 📄 01_data_exploration.ipynb     # EDA and feature analysis
│   ├── 📄 02_preprocessing.ipynb        # Data cleaning & engineering
│   ├── 📄 03_baseline_models.ipynb      # Initial model experiments
│   ├── 📄 04_model_tuning.ipynb         # Hyperparameter optimization
│   └── 📄 05_final_model.ipynb          # Final model & explainability
│
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 data_preprocessing.py         # Data cleaning functions
│   ├── 📄 feature_engineering.py        # Feature creation
│   ├── 📄 model_training.py             # Model training utilities
│   ├── 📄 evaluation.py                 # Evaluation metrics & plots
│   └── 📄 utils.py                      # Helper functions
│
├── 📁 models/
│   ├── 📄 best_model.pkl                # Saved final model
│   └── 📄 model_performance.json        # Performance metrics
│
├── 📁 reports/
│   ├── 📄 final_report.pdf              # Competition submission PDF
│   ├── 📄 figures/                      # Generated plots & charts
│   │   ├── 📄 feature_importance.png
│   │   ├── 📄 correlation_matrix.png
│   │   ├── 📄 roc_curves.png
│   │   └── 📄 shap_summary.png
│   └── 📄 presentation/                 # Presentation materials
│
├── 📁 config/
│   └── 📄 params.yaml                   # Hyperparameters & settings
│
└── 📁 docs/
    ├── 📄 data_dictionary.md            # Feature selection notes
    ├── 📄 medical_vs_non_medical.md     # Feature categorization
    └── 📄 decisions.md                  # Key decisions & justifications
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Method 1: Using pip
```bash
# Clone the repository
git clone https://github.com/ishiniayodyapiumandhi/Enigma.git


# Install dependencies
pip install -r requirements.txt
```

### Method 2: Using Conda
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate modelx-dementia
```

### Dependencies
Key Python packages used:
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.0.0` - Machine learning
- `xgboost>=1.5.0` - Gradient boosting
- `matplotlib>=3.5.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualizations
- `jupyter>=1.0.0` - Notebook environment
- `shap>=0.40.0` - Model explainability
- `imbalanced-learn>=0.8.0` - Handling class imbalance

## 🚀 Usage

### Running the Analysis
1. **Data Exploration**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Preprocessing Pipeline**
   ```bash
   jupyter notebook notebooks/02_preprocessing.ipynb
   ```

3. **Model Training**
   ```bash
   jupyter notebook notebooks/03_baseline_models.ipynb
   ```

4. **Hyperparameter Tuning**
   ```bash
   jupyter notebook notebooks/04_model_tuning.ipynb
   ```

5. **Final Model & Explainability**
   ```bash
   jupyter notebook notebooks/05_final_model.ipynb
   ```

### Using Source Code
```python
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer

# Initialize components
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
trainer = ModelTrainer()

# Build complete pipeline
pipeline = trainer.build_pipeline(preprocessor, feature_engineer)
```

## 🔬 Methodology

### 1. Data Preprocessing
- Missing value imputation
- Categorical variable encoding
- Feature scaling and normalization
- Handling class imbalance

### 2. Feature Engineering
- Creation of new features from existing non-medical data
- Feature selection based on domain knowledge and statistical analysis
- Handling of borderline features with proper justification

### 3. Model Development
**Algorithms Implemented:**
- Logistic Regression (Baseline)
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machines
- Gradient Boosting Machines

### 4. Model Evaluation
**Metrics Used:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix Analysis
- Cross-validation performance

### 5. Explainability
- SHAP values for feature importance
- Partial dependence plots
- Model decision interpretation

## 📈 Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

### Key Insights
- Top non-medical risk factors identified
- Model interpretability and business implications
- Limitations and ethical considerations

## 👥 Team

**Team Name:** Enigma



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- National Alzheimer's Coordinating Center (NACC) for the dataset
- ModelX Hackathon organizers
- IEEE Computational Intelligence Society
- Information Institute of Technology
