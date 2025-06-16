## 🚀 Features

- Detects fraudulent transactions using supervised machine learning models
- Compares Logistic Regression, Random Forest, and CNN performance
- Includes data preprocessing, feature engineering, and hyperparameter tuning
- Uses SMOTE to address class imbalance
- Visualizes key metrics like precision, recall, F1-score, and ROC-AUC

## 📊 Technologies Used

- **Python**
  - `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`
- **ML Models**
  - Logistic Regression
  - Random Forest Classifier
  - Convolutional Neural Network (1D CNN)
- **Tools**
  - Google Colab with T4 GPU
  - SMOTE for oversampling
  - GridSearchCV & KerasTuner for hyperparameter tuning

## 🧪 Results

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 35%      | 1.00      | 0.35   | 0.52     | 0.86    |
| Random Forest       | 100%     | 1.00      | 1.00   | 1.00     | 0.999   |
| CNN (1D)            | 99%      | 1.00      | 0.57   | 0.72     | 0.85    |

🔍 **Insight:** Random Forest outperformed all other models in every metric with proper tuning.

## ⚙️ How to Use

1. Open the CSCI323_Project.ipynb notebook in Jupyter or Colab.

2. Run the notebook cell by cell to:
   - Load and preprocess the data
   - Train and evaluate all three models
   - Visualize performance metrics

## 📌 Limitations
Class imbalance in the dataset (only 0.12% are fraud cases)
CNN model could benefit from further tuning and architectural optimization
Model currently tested on a synthetic dataset (not real financial data)

## 💡 Future Enhancements
Integrate real time fraud alert systems
Experiment with ensemble and deep learning models (e.g., LSTM, Autoencoders)
Deploy in a cloud environment (e.g., AWS Lambda or GCP Functions)

## 📄 License
This project is for educational purposes only. Not intended for production without further validation and regulatory compliance.
