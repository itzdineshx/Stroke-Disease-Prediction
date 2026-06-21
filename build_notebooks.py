import nbformat as nbf
import os

os.makedirs('notebook', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# ---------------------------------------------------------
# Notebook 1: Data Science
# ---------------------------------------------------------
nb1 = nbf.v4.new_notebook()

nb1.cells = [
    nbf.v4.new_markdown_cell("# Data Science: Exploratory Data Analysis & Preparation\nIn this notebook, we'll explore the healthcare-dataset-stroke-data, analyze the distributions, and perform initial data cleaning."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set plotting style\nsns.set_theme(style='whitegrid')"),
    nbf.v4.new_markdown_cell("## 1. Load the Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')\ndf.head()"),
    nbf.v4.new_markdown_cell("## 2. Basic Information & Missing Values"),
    nbf.v4.new_code_cell("df.info()"),
    nbf.v4.new_code_cell("df.describe()"),
    nbf.v4.new_code_cell("# Convert 'N/A' strings in 'bmi' to actual NaNs\ndf['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')\n\n# Check missing values\ndf.isnull().sum()"),
    nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis"),
    nbf.v4.new_code_cell("plt.figure(figsize=(6, 4))\nsns.countplot(data=df, x='stroke')\nplt.title('Distribution of Stroke Cases')\nplt.show()"),
    nbf.v4.new_code_cell("plt.figure(figsize=(10, 5))\nsns.histplot(data=df, x='age', hue='stroke', kde=True)\nplt.title('Age Distribution by Stroke')\nplt.show()"),
    nbf.v4.new_code_cell("plt.figure(figsize=(10, 5))\nsns.histplot(data=df, x='bmi', hue='stroke', kde=True)\nplt.title('BMI Distribution by Stroke')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 4. Data Cleaning"),
    nbf.v4.new_code_cell("# Drop 'id' column as it's not a useful feature\ndf.drop('id', axis=1, inplace=True)\n\n# Impute missing 'bmi' values with the median (or handle differently based on deeper analysis)\nbmi_median = df['bmi'].median()\ndf['bmi'].fillna(bmi_median, inplace=True)\n\n# Remove 'Other' from gender to avoid small categories\ndf = df[df['gender'] != 'Other']\n\nprint(\"Missing values after cleaning:\")\nprint(df.isnull().sum())"),
    nbf.v4.new_markdown_cell("## 5. Save Cleaned Data"),
    nbf.v4.new_code_cell("cleaned_data_path = '../data/processed/stroke-data-cleaned.csv'\ndf.to_csv(cleaned_data_path, index=False)\nprint(f\"Cleaned data saved to {cleaned_data_path}\")")
]

nbf.write(nb1, 'notebook/01_Data_Science.ipynb')

# ---------------------------------------------------------
# Notebook 2: Machine Learning
# ---------------------------------------------------------
nb2 = nbf.v4.new_notebook()

nb2.cells = [
    nbf.v4.new_markdown_cell("# Machine Learning: Feature Engineering, Model Training & Evaluation\nIn this notebook, we prepare features, train classification models, evaluate their performance, and save the best model for deployment."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score\nfrom xgboost import XGBClassifier\n\n# Optional: imblearn for SMOTE if installed\n# from imblearn.over_sampling import SMOTE"),
    nbf.v4.new_markdown_cell("## 1. Load Cleaned Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/stroke-data-cleaned.csv')\ndf.head()"),
    nbf.v4.new_markdown_cell("## 2. Feature Engineering & Preprocessing Setup"),
    nbf.v4.new_code_cell("X = df.drop('stroke', axis=1)\ny = df['stroke']\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nprint(f\"Training set shape: {X_train.shape}\")\nprint(f\"Test set shape: {X_test.shape}\")"),
    nbf.v4.new_code_cell("# Define categorical and numerical columns\ncategorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']\nnumerical_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']\n\n# Create preprocessing steps\nnumeric_transformer = StandardScaler()\ncategorical_transformer = OneHotEncoder(handle_unknown='ignore')\n\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('num', numeric_transformer, numerical_cols),\n        ('cat', categorical_transformer, categorical_cols)\n    ])"),
    nbf.v4.new_markdown_cell("## 3. Model Training & Evaluation (Random Forest)"),
    nbf.v4.new_code_cell("# Create a pipeline with preprocessor and Random Forest\nrf_pipeline = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))\n])\n\n# Train the model\nrf_pipeline.fit(X_train, y_train)\n\n# Predict and evaluate\ny_pred_rf = rf_pipeline.predict(X_test)\nprint(\"Random Forest Classification Report:\")\nprint(classification_report(y_test, y_pred_rf))\nprint(f\"ROC-AUC Score: {roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1]):.4f}\")"),
    nbf.v4.new_markdown_cell("## 4. Model Training & Evaluation (XGBoost)"),
    nbf.v4.new_code_cell("# Create a pipeline with XGBoost\nxgb_pipeline = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42))\n])\n\n# Train\nxgb_pipeline.fit(X_train, y_train)\n\n# Predict and evaluate\ny_pred_xgb = xgb_pipeline.predict(X_test)\nprint(\"XGBoost Classification Report:\")\nprint(classification_report(y_test, y_pred_xgb))\nprint(f\"ROC-AUC Score: {roc_auc_score(y_test, xgb_pipeline.predict_proba(X_test)[:, 1]):.4f}\")"),
    nbf.v4.new_markdown_cell("## 5. Save the Best Model"),
    nbf.v4.new_code_cell("# XGBoost usually performs well, or we can save Random Forest. Let's save Random Forest for this example.\n# (In practice, choose the one with better metrics based on your goals).\nmodel_path = '../models/stroke_prediction_pipeline.pkl'\njoblib.dump(rf_pipeline, model_path)\nprint(f\"Model pipeline saved to {model_path}\")")
]

nbf.write(nb2, 'notebook/02_Machine_Learning.ipynb')
print("Notebooks created successfully!")
