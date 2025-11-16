# House Price Prediction

A machine learning project to predict house prices using the Kaggle "House Prices - Advanced Regression Techniques" dataset. This project demonstrates data exploration, preprocessing, feature engineering, model training, and evaluation. It serves as a portfolio piece showcasing skills in Python, pandas, scikit-learn, and visualization libraries.

## Project Overview
The goal is to predict the sale prices of houses in Ames, Iowa, based on 79 features describing aspects like location, size, quality, and condition. The dataset includes a training set with 1460 samples and a test set with 1459 samples.

- **Dataset**: From Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
- **Approach**: Exploratory Data Analysis (EDA), data preprocessing (handling missing values, encoding, scaling), model training (Gradient Boosting, XGBoost), and evaluation using RMSE and R².
- **Key Results**: 
  - Gradient Boosting achieved a mean RMSE of 0.12 on cross-validation.
  - XGBoost achieved a mean RMSE of 0.13 on cross-validation.
  - Final submission file generated for Kaggle evaluation.

This project highlights end-to-end ML workflow, feature selection using LassoCV, and hyperparameter tuning with GridSearchCV.

## Project Structure
- `data/`: Contains raw (train.csv, test.csv) and processed data files.
- `notebooks/`:
  - `01_eda.ipynb`: Exploratory Data Analysis, including data visualization, correlation, ANOVA for categorical features, outlier detection, and skewness analysis.
  - `02_preprocessing.ipynb`: Data cleaning, missing value imputation using SimpleImputer, categorical encoding with OneHotEncoder, scaling with StandardScaler, and feature selection with LassoCV.
  - `03_modeling.ipynb`: Model training and hyperparameter tuning using Random Forest, Gradient Boosting, Ridge, and XGBoost.
  - `04_evaluation.ipynb`: Cross-validation, RMSE calculation, model selection, and prediction on test data.
- `models/`: Saved models (e.g., best_GradientBoosting.pkl) and CV results.
- `README.md`: This file.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with: pandas numpy matplotlib seaborn scikit-learn xgboost joblib)

## Usage
1. Run EDA: Open `01_eda.ipynb` in Jupyter and execute cells.
2. Run Preprocessing: Open `02_preprocessing.ipynb` to clean and prepare data.
3. Train Models: Open `03_modeling.ipynb` for model training and tuning.
4. Evaluate and Predict: Open `04_evaluation.ipynb` to evaluate models and generate submission.csv.

## Key Insights
- EDA revealed high skewness in `SalePrice`, addressed with log transformation in preprocessing.
- Top numerical features by correlation: OverallQual (0.79), GrLivArea (0.71), etc.
- Top categorical features by eta-squared: Neighborhood (0.55), ExterQual (0.48), etc.
- Gradient Boosting outperformed XGBoost with lower RMSE.

## Future Improvements
- Advanced feature engineering (e.g., interaction terms).
- Ensemble models (e.g., stacking XGBoost and Gradient Boosting).
- Hyperparameter tuning with Optuna for better efficiency.

## About Me
- Name: Hadi (Data ScientistHadi)
- GitHub: [github.com/DataScientistHadi](https://github.com/DataScientistHadi)
- Email: fardmohammadhadi@gmail.com

This project is open for contributions. Feel free to fork and submit pull requests!