import nbformat as nbf
import os

os.makedirs('notebook', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# ---------------------------------------------------------
# Notebook 1: Advanced Data Science
# ---------------------------------------------------------
nb1 = nbf.v4.new_notebook()

nb1.cells = [
    nbf.v4.new_markdown_cell("# Advanced Data Science: Deep Exploratory Data Analysis & Preparation\nThis notebook delves deeply into the dataset, exploring multi-dimensional relationships, identifying outliers, and rigorously preparing the data."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set plotting style\nsns.set_theme(style='darkgrid', palette='muted')\nplt.rcParams['figure.figsize'] = (10, 6)"),
    nbf.v4.new_markdown_cell("## 1. Load the Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')\ndf.head()"),
    nbf.v4.new_markdown_cell("## 2. Basic Information & Missing Values"),
    nbf.v4.new_code_cell("df.info()\nprint(\"\\nMissing values:\")\nprint(df.isnull().sum())"),
    nbf.v4.new_code_cell("# Convert 'N/A' strings in 'bmi' to actual NaNs\ndf['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')\n\n# Visualize missing data\nplt.figure(figsize=(8, 4))\nsns.heatmap(df.isnull(), cbar=False, cmap='viridis')\nplt.title('Missing Data Map')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 3. Advanced Exploratory Data Analysis"),
    nbf.v4.new_code_cell("### Target Distribution (Imbalance)\nplt.figure(figsize=(6, 6))\ncounts = df['stroke'].value_counts()\nplt.pie(counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%', colors=['#4C72B0', '#C44E52'], explode=[0, 0.1])\nplt.title('Stroke Class Imbalance')\nplt.show()"),
    nbf.v4.new_code_cell("### Correlation Heatmap\nnumeric_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']\ncorr = df[numeric_cols].corr()\nplt.figure(figsize=(8, 6))\nsns.heatmap(corr, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\nplt.title('Correlation Matrix of Numerical Features')\nplt.show()"),
    nbf.v4.new_code_cell("### Pairplot of Continuous Variables\ncont_cols = ['age', 'avg_glucose_level', 'bmi', 'stroke']\nsns.pairplot(df[cont_cols].dropna(), hue='stroke', palette='husl', corner=True)\nplt.suptitle('Pairplot of Continuous Features by Stroke', y=1.02)\nplt.show()"),
    nbf.v4.new_code_cell("### Categorical Features Analysis\ncat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']\nfig, axes = plt.subplots(2, 3, figsize=(18, 10))\naxes = axes.flatten()\nfor i, col in enumerate(cat_features):\n    sns.countplot(data=df, x=col, hue='stroke', ax=axes[i], palette='Set2')\n    axes[i].tick_params(axis='x', rotation=45)\n    axes[i].set_title(f'Stroke Distribution by {col}')\nfig.delaxes(axes[5])\nplt.tight_layout()\nplt.show()"),
    nbf.v4.new_markdown_cell("## 4. Outlier Detection & Handling"),
    nbf.v4.new_code_cell("### Boxplots for Outliers\nfig, axes = plt.subplots(1, 3, figsize=(15, 5))\nsns.boxplot(data=df, y='age', ax=axes[0], color='skyblue')\nsns.boxplot(data=df, y='avg_glucose_level', ax=axes[1], color='lightgreen')\nsns.boxplot(data=df, y='bmi', ax=axes[2], color='salmon')\nplt.suptitle('Outlier Detection with Boxplots')\nplt.show()"),
    nbf.v4.new_code_cell("# We can see significant outliers in avg_glucose_level and bmi.\n# We will use IQR method to cap outliers to reduce their extreme effect without losing data.\ndef cap_outliers(series):\n    Q1 = series.quantile(0.25)\n    Q3 = series.quantile(0.75)\n    IQR = Q3 - Q1\n    lower_bound = Q1 - 1.5 * IQR\n    upper_bound = Q3 + 1.5 * IQR\n    return series.clip(lower=lower_bound, upper=upper_bound)\n\ndf['avg_glucose_level'] = cap_outliers(df['avg_glucose_level'])\ndf['bmi'] = cap_outliers(df['bmi'])"),
    nbf.v4.new_markdown_cell("## 5. Data Cleaning & Imputation"),
    nbf.v4.new_code_cell("# Drop 'id' column\ndf.drop('id', axis=1, inplace=True)\n\n# Impute missing 'bmi' values with median\ndf['bmi'].fillna(df['bmi'].median(), inplace=True)\n\n# Remove 'Other' from gender\ndf = df[df['gender'] != 'Other']\n\nprint(\"Cleaned Dataset Info:\")\ndf.info()"),
    nbf.v4.new_markdown_cell("## 6. Save Cleaned Data"),
    nbf.v4.new_code_cell("cleaned_data_path = '../data/processed/stroke-data-cleaned.csv'\ndf.to_csv(cleaned_data_path, index=False)\nprint(f\"Cleaned data saved to {cleaned_data_path}\")")
]
nbf.write(nb1, 'notebook/01_Data_Science.ipynb')

# ---------------------------------------------------------
# Notebook 2: Advanced Machine Learning
# ---------------------------------------------------------
nb2 = nbf.v4.new_notebook()

nb2.cells = [
    nbf.v4.new_markdown_cell("# Advanced Machine Learning: Modeling & Interpretability\nIn this notebook, we address the severe class imbalance using SMOTE, evaluate state-of-the-art models like XGBoost, LightGBM, and CatBoost, tune hyperparameters, and use SHAP for model interpretability."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\nimport warnings\nwarnings.filterwarnings('ignore')\n\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve\n\n# Imblearn\nfrom imblearn.pipeline import Pipeline as ImbPipeline\nfrom imblearn.over_sampling import SMOTE\n\n# Models\nfrom xgboost import XGBClassifier\nfrom lightgbm import LGBMClassifier\nfrom catboost import CatBoostClassifier\n\n# XAI\nimport shap\nshap.initjs()"),
    nbf.v4.new_markdown_cell("## 1. Load Cleaned Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/processed/stroke-data-cleaned.csv')\ndf.head()"),
    nbf.v4.new_markdown_cell("## 2. Feature Engineering Setup"),
    nbf.v4.new_code_cell("X = df.drop('stroke', axis=1)\ny = df['stroke']\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nprint(f\"Training set shape: {X_train.shape}\")\nprint(f\"Test set shape: {X_test.shape}\")"),
    nbf.v4.new_code_cell("# Define column types\ncategorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']\nnumerical_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']\n\n# Preprocessor\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('num', StandardScaler(), numerical_cols),\n        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)\n    ])"),
    nbf.v4.new_markdown_cell("## 3. Modeling Pipeline with SMOTE"),
    nbf.v4.new_code_cell("def evaluate_model(model, name, X_test, y_test):\n    y_pred = model.predict(X_test)\n    y_prob = model.predict_proba(X_test)[:, 1]\n    \n    print(f\"\\n{'='*40}\\n{name} Evaluation\\n{'='*40}\")\n    print(classification_report(y_test, y_pred))\n    print(f\"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\")\n    \n    # Confusion Matrix\n    cm = confusion_matrix(y_test, y_pred)\n    plt.figure(figsize=(4,3))\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)\n    plt.title(f'{name} Confusion Matrix')\n    plt.xlabel('Predicted')\n    plt.ylabel('Actual')\n    plt.show()"),
    nbf.v4.new_code_cell("### LightGBM Pipeline\nlgbm_pipeline = ImbPipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('smote', SMOTE(random_state=42)),\n    ('classifier', LGBMClassifier(random_state=42, verbose=-1))\n])\nlgbm_pipeline.fit(X_train, y_train)\nevaluate_model(lgbm_pipeline, 'LightGBM with SMOTE', X_test, y_test)"),
    nbf.v4.new_code_cell("### CatBoost Pipeline\ncat_pipeline = ImbPipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('smote', SMOTE(random_state=42)),\n    ('classifier', CatBoostClassifier(verbose=0, random_state=42))\n])\ncat_pipeline.fit(X_train, y_train)\nevaluate_model(cat_pipeline, 'CatBoost with SMOTE', X_test, y_test)"),
    nbf.v4.new_code_cell("### XGBoost Pipeline with Hyperparameter Tuning\n# To save time, we do a limited grid search\nxgb_pipeline = ImbPipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('smote', SMOTE(random_state=42)),\n    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))\n])\n\nparam_grid = {\n    'classifier__max_depth': [3, 5],\n    'classifier__n_estimators': [100, 200],\n    'classifier__learning_rate': [0.05, 0.1]\n}\n\ngrid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)\ngrid_search.fit(X_train, y_train)\n\nbest_xgb = grid_search.best_estimator_\nprint(f\"Best XGBoost Parameters: {grid_search.best_params_}\")\nevaluate_model(best_xgb, 'Tuned XGBoost with SMOTE', X_test, y_test)"),
    nbf.v4.new_markdown_cell("## 4. Advanced Evaluation Visualizations"),
    nbf.v4.new_code_cell("best_model = best_xgb # Using Tuned XGBoost as our best model\n\n# ROC Curve\ny_prob_best = best_model.predict_proba(X_test)[:, 1]\nfpr, tpr, _ = roc_curve(y_test, y_prob_best)\n\nplt.figure(figsize=(7, 5))\nplt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob_best):.3f})')\nplt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('Receiver Operating Characteristic')\nplt.legend(loc=\"lower right\")\nplt.show()"),
    nbf.v4.new_markdown_cell("## 5. Model Interpretability with SHAP"),
    nbf.v4.new_code_cell("# Extract preprocessor and feature names\npreprocessor_fit = best_model.named_steps['preprocessor']\ncat_encoder = preprocessor_fit.named_transformers_['cat']\ncat_features_encoded = cat_encoder.get_feature_names_out(categorical_cols)\nfeature_names = numerical_cols + list(cat_features_encoded)\n\n# Transform X_test to get data for SHAP\nX_test_transformed = preprocessor_fit.transform(X_test)\nX_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)\n\n# Initialize SHAP explainer on the XGBoost classifier step\nxgb_classifier = best_model.named_steps['classifier']\nexplainer = shap.TreeExplainer(xgb_classifier)\nshap_values = explainer.shap_values(X_test_df)\n\n# SHAP Summary Plot\nplt.figure(figsize=(10, 8))\nshap.summary_plot(shap_values, X_test_df, show=False)\nplt.title('SHAP Summary Plot - Feature Impact on Stroke Prediction')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 6. Save the Final Advanced Model"),
    nbf.v4.new_code_cell("model_path = '../models/stroke_prediction_pipeline.pkl'\njoblib.dump(best_model, model_path)\nprint(f\"Advanced model pipeline saved to {model_path}\")")
]
nbf.write(nb2, 'notebook/02_Machine_Learning.ipynb')
print("Advanced Notebooks created successfully!")
